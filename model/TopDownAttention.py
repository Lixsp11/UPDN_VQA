import torch
import torch.nn
import torchtext
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm


class TDAttention(torch.nn.Module):
    def __init__(self, num_token, w_dim=300, v_dim=2048, hid_dim=512, N=3129):
        """
        
        :param w_dim: Word embedding dim.
        :param v_dim: Feature vector dim.
        :param hid_dim: Hidden dim of GRU.
        :param N: Num of candidate answers.
        """
        super(TDAttention, self).__init__()
        # Word embedding
        self.embd = WordEmbedding(w_dim, num_token)
        # Question Embedding
        self.gru = torch.nn.GRU(input_size=w_dim, hidden_size=hid_dim, 
                                num_layers=1, bidirectional=False, batch_first=True)
        # Image Attention
        self.v_proj = FCNet(v_dim, hid_dim)
        self.q_proj = FCNet(hid_dim, hid_dim)
        self.dropout1 = torch.nn.Dropout(0.2)
        self.linear1 = weight_norm(torch.nn.Linear(hid_dim, 1), dim=None)
        # Multimodal Fusion
        self.fq = FCNet(hid_dim, hid_dim)
        self.fv = FCNet(v_dim, hid_dim)
        # Output Classifier
        self.classifier = SimpleClassifier(hid_dim, hid_dim * 2, N)


    def forward(self, v, q):
        """
        
        :param v: Feature vector, shape=[B, K, D]
        :param q: Tokenized one-hot word vector, shape=[B, L, N + 1]
        :return : Predicted probability distribution, shape=[B, N]
        """
        # Word embedding
        q = self.embd(q)
        # Question Embedding
        self.gru.flatten_parameters()
        output, _ = self.gru(q)
        q = output[:, -1]  # shape=[B, H]
        # Image Attention
        v = self.v_proj(v)
        q = self.q_proj(q).broadcast_to(*v.shape[:2], q.shape[-1])
        a = self.linear1(self.dropout1(v * q))
        a = torch.softmax(a, dim=-2)  # ?
        v_hat = torch.mul(a, v).sum(dim=-2, keepdim=False)  # [B, D]
        # Multimodal Fusion
        h = self.fq(q) * self.fv(v_hat)  # [B, H]
        # # Output Classifier
        return self.classifier(h)

class WordEmbedding(torch.nn.Module):
    def __init__(self, w_dim, num_token):
        self.embd = torch.nn.Embedding(num_token + 1, w_dim, padding_idx=num_token)
        self.num_token = num_token

    def init_weight(self, weight):
        assert weight.shape == self.embd.weight[1:].shape
        self.embd.weight.data[:self.num_token] = weight


    def forward(self, q):
        """
        :param q: Tokenized query string, List[List[str]]
        :return : Embedding vector of the query string, shape=[L, W]
        """
        return self.embd(q)


class FCNet(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FCNet, self).__init__()

        self.linear = weight_norm(torch.nn.Linear(in_dim, out_dim), dim=None)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x))


class SimpleClassifier(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout=0.5):
        super(SimpleClassifier, self).__init__()
        layers = [
            weight_norm(torch.nn.Linear(in_dim, hid_dim), dim=None),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout, inplace=True),
            weight_norm(torch.nn.Linear(hid_dim, out_dim), dim=None)
        ]
        self.main = torch.nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits
