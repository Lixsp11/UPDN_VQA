import sys
import csv
import tqdm
import torch
import base64
import torchtext
import torch.nn.functional as F


class Tokenizer(object):
    def __init__(self, vocab):
        self.tokenizer = torchtext.data.utils.get_tokenizer("basic_english")
        self.vocab = vocab
    
    def __call__(self, q):
        """
        
        :param q: Query string, str
        :retrun : Tokenized query string, Dict[str]
        """
        q = self.tokenizer(q)
        return [word if word in self.vocab else '[unk]' for word in q]


class WordEmbedding(object):
    def __init__(self, w_dim):
        self.embd = torchtext.vocab.GloVe(name="840B", dim=w_dim)
        self.embd.itos.extend(['<unk>'])
        self.embd.stoi['<unk>'] = self.embd.vectors.shape[0]
        # print("here", flush=True)
        # self.embd.vectors = torch.vstack((self.embd.vectors, torch.zeros(1, w_dim)))
    
    def __call__(self, q):
        """
        :param q: Tokenized query string, Dict[str]
        :return : Embedding vector of the query string, shape=[L, W]
        """
        return self.embd.get_vecs_by_tokens(q)

def VQA2feats(feat_path, feats):
    csv.field_size_limit(sys.maxsize)

    feats.clear()
    fieldnames = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']
    with open(feat_path, 'r', encoding='ascii') as f:
        reader = csv.DictReader(f, delimiter='\t', fieldnames=fieldnames)
        for item in tqdm.tqdm(reader, desc="feat", file=sys.stdout):
            feat = base64.b64decode(item['features'])
            feat = torch.frombuffer(feat, dtype=torch.float32).reshape((36, -1))
            feat = F.normalize(feat, dim=-1)
            feats[item['image_id']] = feat
            