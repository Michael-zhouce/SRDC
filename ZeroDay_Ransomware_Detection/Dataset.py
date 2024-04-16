import torch
from transformers import GPT2Model, GPT2Tokenizer
import pandas as pd
import numpy as np

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.padding_side = "right"
tokenizer.pad_token = tokenizer.eos_token
labels = {
        "Goodware": 0,
        "malware": 1,
            }

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataframe):
        dataframe= dataframe.fillna('')
        texts = dataframe[['apiFeatures', 'dropFeatures', 'regFeatures', 'filesFeatures', 'filesEXTFeatures', 'dirFeatures', 'strFeatures']].values
        new_array = []  
        for sublist in texts:  
             token_text =  tokenizer(sublist.tolist(),padding='max_length',max_length=1024,truncation=True,return_tensors="pt")
             new_array.append(token_text)  
        self.texts = new_array
        self.labels = np.where((dataframe['family'] >= 1) & (dataframe['family'] <= 11), 1, dataframe['family'])
        assert len(self.texts) == len(self.labels), '[ERROR] texts count not equal label count.'

    def __getitem__(self, idx):
        text= self.texts[idx]
        label = self.labels[idx]
        return text,  label
        
    def classes(self):
        return self.labels
    
    def __len__(self):
        return len(self.labels)