import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from transformers import RobertaTokenizer
import torch
import csv

class MyDataSet(Dataset):
    def __init__(self, filename, maxlen):

        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True)

        df = pd.read_csv(filename,sep='\t',
                quotechar='"',
                engine='python', 
                quoting=csv.QUOTE_NONE,
                escapechar='\\',
                keep_default_na=False,
                dtype={'index':np.int32,'text':str,'V':np.float64, 'A':np.float64})
        
        
        self.texts = df['text'].to_list()
        self.valence = df['V'].to_list()
        self.arousal = df['A'].to_list()
        self.maxlen = maxlen

    def __getitem__(self, idx):
        item = { }
        aux = self.tokenizer(self.texts[idx], max_length=self.maxlen, truncation=True, padding=False)
        item['input_ids'] = torch.tensor(aux['input_ids'])
        item['attention_mask'] = torch.tensor(aux['attention_mask'])
        item['labels'] = torch.tensor( [ self.valence[idx], self.arousal[idx] ] )

        return item

    def __len__(self):
        return len(self.texts)
    