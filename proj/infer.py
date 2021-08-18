from tqdm import tqdm

import torch
from torch.nn.functional import softmax
from CustomDataset import BERTDataset

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Model Inference
def predict(texts: list, tok, model, device=device):
    ### argument
    max_len=64
    batch_size=8
    ###

    preds, scores = [], []
    data_infer = BERTDataset(dataset=texts, sent_idx=0, label_idx=None, bert_tokenizer=tok, max_len=max_len, pad=True, pair=False)
    infer_dataloader = torch.utils.data.DataLoader(data_infer, batch_size=batch_size, num_workers=1)

    model.eval()
    for batch_id, (token_ids, valid_length, segment_ids) in enumerate(infer_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        with torch.no_grad():    
            logits = model(token_ids, valid_length, segment_ids)
        pred = torch.argmax(logits, axis=1).cpu().numpy()
        score = softmax(logits, dim=1)
        preds.extend(pred)
        scores.extend(score)
        del logits
    torch.cuda.empty_cache()
    return preds, scores