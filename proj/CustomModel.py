import torch.nn as nn

from transformers import BertModel, BertPreTrainedModel


class SentimentBertModel(BertPreTrainedModel):
    def __init__(self, config, **kargs):
        config.update(kargs)
        super(SentimentBertModel, self).__init__()

        self.bert = BertModel(config)
        self.drop = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LeakyReLU(0.1),
            nn.Linear(config.hidden_size, self.config.n_classes)
        )

    def forward(
            self,
            input_ids,
            attention_mask
        ):

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        sequence_output = outputs[0]
        logits = self.classifier(sequence_output[:, 0])

        return logits
