# models/text_model.py

import torch.nn as nn
from transformers import BertModel

class TextModel(nn.Module):
    def __init__(self, pretrained_model_name='bert-base-uncased', dropout=0.3):
        super(TextModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(self.bert.config.hidden_size, 128)  # Output size for fusion
    
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        output = self.dropout(pooled_output)
        return self.out(output)
