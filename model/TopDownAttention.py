import torch
import numpy as np
import torch.nn.functional as F


class TDAttention(torch.nn.Module):
    def __init__(self, embd_weight_path, w_dim=300, v_dim=2048, hid_dim=512, N=3129):
        """
        
        :param embd_weight_path: Path of pretrained word embdeeing weight
        :param w_dim: Word embedding dim.
        :param v_dim: Feature vector dim.
        :param hid_dim: Hidden dim of GRU.
        :param N: Num of candidate answers.
        """
        super(TDAttention, self).__init__()
        # Word embedding
        self.embd = WordEmbedding(embd_weight_path)
        # Question Embedding
        self.gru = torch.nn.GRU(input_size=w_dim, hidden_size=hid_dim, num_layers=1, bidirectional=False, batch_first=True)
        # Image Attention
        self.fa = GatedTanh(v_dim + hid_dim, hid_dim)
        self.linear1 = torch.nn.Linear(hid_dim, 1)
        # Multimodal Fusion
        self.fq = GatedTanh(hid_dim, hid_dim)
        self.fv = GatedTanh(v_dim, hid_dim)
        # Output Classifier
        self.fo_text = GatedTanh(hid_dim, w_dim)
        self.fo_img = GatedTanh(hid_dim, v_dim)
        self.linear2 = torch.nn.Linear(w_dim, N)
        self.linear3 = torch.nn.Linear(v_dim, N)


    def forward(self, v, q):
        """
        
        :param v: Feature vector, shape=[B, K, D]
        :param q: Tokenized question vector, shape=[B, L]
        :return : Predicted probability distribution, shape=[B, N]
        """
        # Word embedding
        q = self.embd(q)
        # Question Embedding
        self.gru.flatten_parameters()
        _, hidden = self.gru(q)  # hidden.shape=[1, B, H]
        q = hidden.transpose(0, 1)  # [B, 1, H]
        # Image Attention
        v = F.normalize(v, dim=-1)
        a = torch.concat((v, q.broadcast_to(*v.shape[:2], q.shape[-1])), dim=-1)
        a = self.linear1(self.fa(a))  # [B, K, 1]
        a = torch.softmax(a, dim=-2)  # [B, K, 1]
        v_hat = torch.mul(a, v).sum(dim=-2, keepdim=False)  # [B, D]
        # Multimodal Fusion
        h = self.fq(q.squeeze(1)) * self.fv(v_hat)  # [B, H]
        # Output Classifier
        y = self.linear2(self.fo_text(h)) + self.linear3(self.fo_img(h))
        if self.training:
            return y
        else:
            return torch.sigmoid(y)


class WordEmbedding(torch.nn.Module):
    def __init__(self, weight_path):
        super(WordEmbedding, self).__init__()

        weight = torch.from_numpy(np.load(weight_path))
        weight = torch.vstack((weight, torch.zeros(weight.shape[1])))
        self.embd = torch.nn.Embedding.from_pretrained(weight, freeze=False, 
                                                        padding_idx=weight.shape[0] - 1)

    def forward(self, q):
        """
        :param q: Tokenized query string, List[List[str]]
        :return : Embedding vector of the query string, shape=[L, W]
        """
        return self.embd(q)


class GatedTanh(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GatedTanh, self).__init__()
        self.linear1 = torch.nn.Linear(in_dim, out_dim, bias=True)
        self.linear2 = torch.nn.Linear(in_dim, out_dim, bias=True)
        self.dropout = torch.nn.Dropout(p=0.2)
    
    def forward(self, x):
        y_hat = torch.tanh(self.linear1(x))
        g = torch.sigmoid(self.linear2(x))
        y = torch.mul(y_hat, g)
        return self.dropout(y)