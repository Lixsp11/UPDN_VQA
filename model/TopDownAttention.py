import torch

class TDAttention(torch.nn.Module):
    def __init__(self, w_dim=300, v_dim=2048, hid_dim=512, N=3129):
        """
        
        :param w_dim: Word embedding dim.
        :param v_dim: Feature vector dim.
        :param hid_dim: Hidden dim of GRU.
        :param N: Num of candidate answers.
        """
        super(TDAttention, self).__init__()
        # Question Embedding
        self.gru = torch.nn.GRU(input_size=w_dim, hidden_size=hid_dim, num_layers=1, bidirectional=False, batch_first=True)
        # Image Attention
        self.fa = GatedTanh(max(v_dim, hid_dim), hid_dim)
        self.linear1 = torch.nn.Linear(hid_dim, 1, bias=False)
        # Multimodal Fusion
        self.fq = GatedTanh(hid_dim, hid_dim)
        self.fv = GatedTanh(v_dim, hid_dim)
        # Output Classifier
        self.fo_text = GatedTanh(hid_dim, w_dim)
        self.fo_img = GatedTanh(hid_dim, v_dim)
        self.linear2 = torch.nn.Linear(w_dim, N, bias=False)
        self.linear3 = torch.nn.Linear(v_dim, N, bias=False)

    def forward(self, v, q):
        """
        
        :param v: Feature vector, shape=[B, K, D]
        :param q: Question embedding vector, shape=[B, L, H]
        :return : Predicted probability distribution, shape=[B, N]
        """
        # Question Embedding
        _, hidden = self.gru(q)  # hidden.shape=[1, B, H]
        q = hidden.reshape(hidden.shape[1], 1, -1)  # [B, 1, H]
        # Image Attention
        a = torch.vstack(v, q)
        a = self.linear1(self.fa(a))  # [B, K, 1]
        a = torch.softmax(a, dim=-2)  # [B, K, 1]
        v_hat = torch.mul(a, v).sum(dim=-2, keepdim=False)  # [B, D]
        # Multimodal Fusion
        h = torch.mul(self.fq(q.squeeze(1)), self.fv(v_hat))  # [B, H]
        # Output Classifier
        return self.linear2(self.fo_text(h)) + self.linear3(self.fo_img(h))  # [B, N]

class GatedTanh(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GatedTanh, self).__init__()
        self.linear1 = torch.nn.Linear(in_dim, out_dim, bias=True)
        self.linear2 = torch.nn.Linear(in_dim, out_dim, bias=True)
        self.dropout = torch.nn.Dropout()
    
    def forward(self, x):
        y_hat = torch.tanh(self.linear1(x))
        g = torch.sigmoid(self.linear2(x))
        y = torch.mul(y_hat, g)
        return self.dropout(y)
