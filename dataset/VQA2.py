import sys
import json
import tqdm
import torch
import numpy as np
import torch.utils.data
from .utils import Tokenizer, WordEmbedding


class VQA2Dataset(torch.utils.data.Dataset):
    def __init__(self, feats, qu_path, ann_path, word_dim, ann_num_classes):
        super().__init__()
        with open(qu_path, "r") as f:
            qus = json.load(f)
        with open(ann_path, "r") as f:
            anns = json.load(f)

        word_embd = WordEmbedding(word_dim)
        tokenizer = Tokenizer(word_embd.embd.stoi.keys())

        qus = qus['questions']
        # qu example: {"image_id": 458752, "question": "Is this man a professional baseball player?", "question_id": 458752003}
        # ann example: {"question_id": 458752003, "image_id": 458752, "labels": [3, 9], "scores": [1, 0.3]}
        self.imgs, self.qus, self.anns = [], [], []
        for index in tqdm.trange(len(qus), desc="info", file=sys.stdout): 
            if qus[index]['question_id'] != anns[index]['question_id']:
                raise RuntimeError("question_id  and ann_id not match.")
            
            self.imgs.append(qus[index]['image_id'])
            self.qus.append(word_embd(tokenizer(qus[index]['question'])))
            ann = torch.zeros(ann_num_classes)
            ann[anns[index]['labels']] = torch.tensor(anns[index]['scores']).float()
            self.anns.append(ann)
        self.feats = feats
        self.imgs = np.array(self.imgs)
        self.qus = np.array(self.qus)
        self.anns = torch.stack(self.anns)
    
    def __getitem__(self, index):
        return self.feats[str(self.imgs[index])], self.qus[index], self.anns[index]

    def __len__(self):
        return len(self.qus)