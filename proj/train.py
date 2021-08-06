import os, sys
import logging
import argparse
import csv
import gc
from tqdm import tqdm
from collections import OrderedDict

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import BertTokenizer, BertModel
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import WEIGHTS_NAME, CONFIG_NAME

from CustomDataset import MovieDataset
from CustomModel import SentimentBertModel
from util.log import setup_default_logging
from util.metrics import AverageMeter
from util.collate import customCollate

try:
    import wandb
    has_wandb = True
except ImportError: 
    has_wandb = False

_logger = logging.getLogger('train')
parser = argparse.ArgumentParser(description='Train Config', add_help=False)

def get_dataset(data_path, tokenizer, max_len, random_seed, is_train=True):
    document, target = [], []
    with open(data_path, 'r'):
        lines = csv.reader()
        header = lines.pop(0)
        for line in lines:
            line = line.split('\t')
            document.append(line[1])
            target.append(line[2])
    if is_train:
        train_doc, valid_doc, train_target, valid_target = train_test_split(
            document, target, test_size=0.2, shuffle=True, stratify=target, random_state=random_seed
        )
        train_dataset = MovieDataset(tokenizer, train_doc, train_target, max_len)
        valid_dataset = MovieDataset(tokenizer, valid_doc, valid_target, max_len)
        return train_dataset, valid_dataset
    else:
        test_dataset = MovieDataset(tokenizer, document, target, max_len)
        return test_dataset

def initalize_model(prev_model, max_len, finetune=False):
    model = BertModel.from_pretrained(prev_model)
    model = SentimentBertModel.from_pretrained(prev_model, 
                                                n_classes=2,
                                                max_length=max_len)
    if finetune:
        for param in model.parameters():
            param.requires_grad = True
    else:
        for name, param in model.named_parameters():
            if 'classifier' in name:
                param.requires_grad = True

    return model

def train_one_epoch(model, loader, device, loss_fn, optimizer):
    ### argument
    log_interval = None
    ###

    model.train()
    
    train_loss_m = AverageMeter()
    last_idx = len(loader) - 1
    for idx, batch in tqdm(enumerate(loader), total=len(loader)):
        last_batch = idx == last_idx
        optimizer.zero_grad()    
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(input_ids=batch['input_ids'],
                        token_type_ids=batch['token_type_ids'],
                        attention_mask=batch['attention_mask'])

        loss, _ = loss_fn(logits, batch['label'])
        train_loss_m.update(loss.data.item(), batch['input_ids'].size(0))

        loss.backward()
        optimizer.step()

        if last_batch or (idx+1 % log_interval == 0):
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            avg_lr = sum(lrl)/len(lrl)
            _logger.info(
                f'avg_train_loss : {train_loss_m.avg}, LR : {avg_lr}')

        del batch, loss

    del loader
    gc.collect()

    metrics = OrderedDict([('loss', train_loss_m.avg)])

    return metrics
        
def validation(model, loader, device, loss_fn):
    ### argument
    log_interval = None
    ###

    model.eval()
    val_loss_m = AverageMeter()
    acc_m = AverageMeter()

    last_idx = len(loader) - 1
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            last_batch = idx == last_idx
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(input_ids=batch['input_ids'],
                                 token_type_ids=batch['token_type_ids'],
                                 attention_mask=batch['attention_mask'])
            
            loss, scores = loss_fn(logits, batch['label'])
            val_loss_m.update(loss.data.item(), batch['input_ids'].size(0))
            
            acc = sum([i == j for i, j in zip(torch.argmax(scores, 1).tolist(), batch['label'])]) / len(batch['label'])
            acc_m.update(acc)
            val_loss_m.update(loss, batch['input_ids'].size(0))

            del batch, loss, scores            

        if last_batch or (idx+1 % log_interval == 0):
            _logger.info(
                f'avg_val_loss : {val_loss_m.avg}, avg_accuracy : {acc_m.avg}')

    metrics = OrderedDict([('loss', val_loss_m.avg), ('accuracy', acc_m.avg)])

    del loader
    gc.collect()

    return metrics

def save(model, tokenizer, args):
    _logger.info('saving model...')
    os.makedirs(args.new_model, exist_ok=True)
    torch.save(model.state_dict(), args.new_model + WEIGHTS_NAME)
    model.config.to_json_file(args.new_model + CONFIG_NAME)
    tokenizer.save_pretrained(args.new_model)
    _logger.info(f'saved! {args.new_model}')

def main():
    ### argument
    random_seed = None
    prev_model = None
    train_data_path = None
    test_data_path = None
    max_len = 512
    batch_size = None
    lr = None
    epochs = None
    valid_every_n_batch = None
    save_best = None
    val_metric = None
    ###

    setup_default_logging()
    if torch.cuda.is_available():
        device = torch.device("cuda")
        _logger.info(f'GPU: {torch.cuda.get_device_name(0)}')
    else:
        device = torch.device("cpu")
    torch.manual_seed(random_seed)

    tokenizer = BertTokenizer.from_pretrained(prev_model, do_lower_case=False)
    train_dataset, valid_dataset = get_dataset(train_data_path, tokenizer, max_len, random_seed, is_train=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=customCollate)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=customCollate)

    model = initalize_model(prev_model, max_len)
    model = model.to(device)

    total_steps = len(train_loader) * epochs
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=2, num_training_steps=total_steps)
    loss_fn = nn.CrossEntrophy()

    min_val_loss = 10.0
    max_accuracy = 0.0
    try:
        for epoch in range(epochs):
            gc.collect()
            _logger.info(f'Train {epoch} epoch')
            train_metrics = train_one_epoch(model, train_loader, device, optimizer)

            gc.collect()

            if (epoch + 1) % valid_every_n_batch == 0:
                _logger.info('validation ...')
                valid_metrics = validation(model, valid_loader, device)
                scheduler.step()

                gc.collect()

                # Save / Early Stop
                if save_best.lower().startswith('loss'):
                    _logger.info(f'best loss was {min_val_loss}')
                    if valid_metrics[val_metric] < min_val_loss:
                        min_val_loss = valid_metrics[val_metric]
                        _logger.info(f'best loss changed to {min_val_loss}')
                        save(model, tokenizer)
                elif save_best.lower().startswith('accuracy'):
                    _logger.info(f'best accuracy was {max_accuracy}')                    
                    if valid_metrics[val_metric] > max_accuracy:
                        max_accuracy = valid_metrics[val_metric]
                        _logger.info(f'best accuracy changed to {min_val_loss}')
                        save(model, tokenizer)

    except KeyboardInterrupt:
        if save_best.lower().startswith('accuracy'):
            _logger.info(f'Model accuracy was {max_accuracy}')
        elif save_best.lower().startswith('loss'):
            _logger.info(f'Model valid loss was {min_val_loss}')
        elif save_best.lower().startswith('weight'):
            _logger.info(f'Model weighted valid loss was {min_val_loss}')
        _logger.info('Bye!')

if __name__ == "__main__":
    main()