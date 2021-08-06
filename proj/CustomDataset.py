import csv

import torch
from torch.utils.data import Dataset

class MovieDataset(Dataset):
    def __init__(self, tokenizer, document, target, max_len):
        super().__init__()
        self.tokenizer = tokenizer
        self.document = document
        self.target = target
        self.max_len = max_len

    def __getitem__(self, index):
        label = torch.tensor(self.target[index], dtype=torch.long)
        data = self.tokenizer.encode_plus(
            self.document[index],
            max_length=self.max_len,
            truncation=True,
            pad_to_max_length=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors='pt')
        return data, label