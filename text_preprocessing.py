# preprocessing/text_preprocessing.py

import os
import torch
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset

class TranscriptDataset(Dataset):
    def __init__(self, transcripts, labels, tokenizer, max_length=512):
        self.transcripts = transcripts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.transcripts)

    def __getitem__(self, idx):
        transcript = str(self.transcripts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            transcript,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def create_text_data_loader(transcripts, labels, tokenizer, batch_size, max_length):
    ds = TranscriptDataset(
        transcripts=transcripts,
        labels=labels,
        tokenizer=tokenizer,
        max_length=max_length
    )
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=4)
