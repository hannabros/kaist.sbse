import os, sys
import logging
import argparse

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm

from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from transformers import WEIGHTS_NAME, CONFIG_NAME

from CustomDataset import BERTDataset
from CustomModel import BERTClassifier
from util.log import setup_default_logging

_logger = logging.getLogger('train')
parser = argparse.ArgumentParser(description='Train Config', add_help=False)

def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc

def save(model, e, optimizer, loss, model_path):
    _logger.info('saving model...')
    os.makedirs(model_path, exist_ok=True)
    torch.save({
            'epoch': e,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
            }, model_path + WEIGHTS_NAME)
    _logger.info(f'saved! {model_path}')

def main():
    ### argument
    train_data_path = "/home/ubuntu/workspace/kaist.sbse/proj/data/ratings_train.txt"
    test_data_path = "/home/ubuntu/workspace/kaist.sbse/proj/data/ratings_train.txt"
    save_model_path = "./model/"
    
    ## Setting parameters
    max_len = 64
    batch_size = 64
    warmup_ratio = 0.1
    num_epochs = 5
    max_grad_norm = 1
    log_interval = 200
    learning_rate =  5e-5
    ###

    if torch.cuda.is_available():
        device = torch.device("cuda")
        _logger.info(f'GPU: {torch.cuda.get_device_name(0)}')
    else:
        device = torch.device("cpu")

    bertmodel, vocab = get_pytorch_kobert_model()

    dataset_train = nlp.data.TSVDataset(train_data_path, field_indices=[1,2], num_discard_samples=1)
    dataset_test = nlp.data.TSVDataset(test_data_path, field_indices=[1,2], num_discard_samples=1)

    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
    
    data_train = BERTDataset(dataset_train, 0, 1, tok, max_len, True, False)
    data_test = BERTDataset(dataset_test, 0, 1, tok, max_len, True, False)

    train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=1)
    test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=1)

    model = BERTClassifier(bertmodel,  dr_rate=0.5).to(device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    t_total = len(train_dataloader) * num_epochs
    warmup_step = int(t_total * warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

    for e in range(num_epochs):
        train_acc = 0.0
        test_acc = 0.0
        best_acc = 0.0
        model.train()
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length= valid_length
            label = label.long().to(device)
            out = model(token_ids, valid_length, segment_ids)
            loss = loss_fn(out, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            train_acc += calc_accuracy(out, label)
            if batch_id % log_interval == 0:
                _logger.info("epoch {} batch id {} loss {} train acc {}".format(e+1, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))
        _logger.info("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))
        
        model.eval()
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(test_dataloader)):
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length= valid_length
            label = label.long().to(device)
            out = model(token_ids, valid_length, segment_ids)
            test_acc += calc_accuracy(out, label)
        _logger.info("epoch {} test acc {}".format(e+1, test_acc / (batch_id+1)))

        if test_acc > best_acc:
            _logger.info(f'best accuracy was {best_acc}')
            best_acc = test_acc
            _logger.info(f'best accuracy changed to {best_acc}')
            save(model, e, optimizer, loss, save_model_path)

if __name__ == "__main__":
    setup_default_logging()
    main()