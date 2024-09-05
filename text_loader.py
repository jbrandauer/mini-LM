from pathlib import Path
from typing import Generator
import json
import torch
from torch.utils.data import Dataset, DataLoader
import os

from text_encoding import tokenize

def _chunkify(tokens_list: list[int],
              n_chunks: int,
              offset: int=0)->Generator:
    for i in range(0, len(tokens_list), n_chunks):  
        yield tokens_list[i+offset:i+offset + n_chunks]

class TextData(Dataset):
    def __init__(self, file_path: Path, context_length: int, dtype=torch.float32):
        super(TextData, self).__init__()
        with open(file_path, "r") as f:
            text = f.read()
        text = text[:len(text)]
        (tokenized_text, self.word_to_index, self.index_to_word), self.vocab_size = tokenize(text)
        text_chunks = list(_chunkify(tokenized_text, context_length))
        target_text_chunks = list(_chunkify(tokenized_text, context_length, 1))
        if(Path("text_data.json").exists()):
            os.remove("text_data.json")
        for i, chunk in enumerate(text_chunks):
            with open("text_data.json", "a") as fw:
                if(len(chunk)  == context_length):
                    fw.write(json.dumps({"Seq. ID": i+1,
                                     "Input": chunk, 
                                     "Target": target_text_chunks[i]}) 
                                     + "\n")
        self.text_path = "text_data.json"
        self.dtype = dtype
                
    def __getitem__(self, idx):
        with open(self.text_path, "r") as f:
            lines = f.readlines()
        data = json.loads(lines[idx])
        input = data["Input"]
        target = data["Target"]
        return torch.tensor(input), torch.tensor(target)
    
    def __len__(self):
        with open(self.text_path, "r") as f:
            lines = f.readlines()
        return len(lines)
    

def get_text_data_loader(batch_size: int,
                         context_length: int,
                         file_path: str = "shakespeare.rtf"):
    dataset = TextData(file_path, context_length)
    data_loader = DataLoader(dataset=TextData(file_path, context_length), batch_size=batch_size, shuffle=True)
    return data_loader, dataset.word_to_index, dataset.index_to_word
        


if(__name__=="__main__"):
    data_loader, _, _ = get_text_data_loader(batch_size=4, context_length=10)
    for i, (input, target) in enumerate(data_loader):
        if(i>4):
            break
        print("First batch shape: ", input.shape)
        print("First batch target: ", target.shape)

