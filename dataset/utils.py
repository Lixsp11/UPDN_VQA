import sys
import csv
import tqdm
import torch
import base64
import torchtext
import torch.nn.functional as F


class Tokenizer(object):
    def __init__(self, idx2word, word2idx):
        self.idx2word = idx2word
        self.word2idx = word2idx
        self.num_token = len(idx2word)
    
    def __call__(self, q):
        """
        
        :param q: Query string, str
        :retrun : Tokenized query string, List[str]
        """
        q = q.lower()
        q = q.replace(',', '').replace('?', '').replace('\'s', ' \'s').replace('-', 
                ' ').replace('.', '').replace('"', '').replace('n\'t', ' not').replace('$', ' dollar ')
        q = q.split()
        idxs = [self.word2idx[word] if word in self.idx2word else self.num_token for word in q]
        return torch.tensor(idxs)


def VQA2feats(feat_path, feats):
    csv.field_size_limit(sys.maxsize)

    feats.clear()
    fieldnames = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']
    with open(feat_path, 'r', encoding='ascii') as f:
        reader = csv.DictReader(f, delimiter='\t', fieldnames=fieldnames)
        for item in tqdm.tqdm(reader, desc="feat", file=sys.stdout):
            feat = base64.b64decode(item['features'])
            feat = torch.frombuffer(feat, dtype=torch.float32).reshape((36, -1))
            # feat = F.normalize(feat, dim=-1)
            feats[item['image_id']] = feat
            