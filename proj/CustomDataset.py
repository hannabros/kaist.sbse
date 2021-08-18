from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len, pad, pair):
        transform = nlp.data.BERTSentenceTransform(bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        if label_idx is not None:
            self.sentences = [transform([i[sent_idx]]) for i in dataset]
            self.labels = [np.int32(i[label_idx]) for i in dataset]
        else:
            self.sentences = [transform([i]) for i in dataset]
            self.labels = None

    def __getitem__(self, i):
        if self.labels is not None:
            return (self.sentences[i] + (self.labels[i], ))
        else:
            return (self.sentences[i])

    def __len__(self):
        return (len(self.sentences))
