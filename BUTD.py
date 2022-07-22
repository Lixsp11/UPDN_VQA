from turtle import forward
import torch
import torchvision
import torchtext.vocab


class BUTD(torch.nn.Module):
    def __init__(self, bu_model, td_model):
        super(BUTD, self).__init__()
        self.bu_model = bu_model
        self.td_model = td_model

    def forward(self, img, qu):
        """

        :param img: Image, shape=[B, C, H, W].
        :param qu: Questions, shape=[B, 14].
        :return: Predicted scores of candidate answers.
        """
        with torch.no_grad():
            v = self.bu_model(img)
        return self.td_model(qu, v)


class BU(torch.nn.Module):
    def __init__(self, K=36):
        """

        :param K: Num of features.
        """
        super(BU, self).__init__()
        self.preprocess = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1.transforms()
        self.rcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
            weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1,
            box_detections_per_img=K
        )
        self.rcnn.backbone.body.layer4.v = None

        def v_hook(self, _, _v):
            self.v = _v

        self.rcnn.backbone.body.layer4.register_forward_hook(v_hook)

    def forward(self, img):
        """

        :param img: Image, shape=[B, C, H, W].
        :return: Features in a List, shape=[B, K, D] (D=2048 with ResNet backbone).
        """
        img = self.preprocess(img)
        prediction = self.rcnn(img)
        boxes = [x["boxes"] for x in prediction]
        boxes = torch.nn.utils.rnn.pad_sequence(boxes, batch_first=True, padding_value=0)
        # map img box to feature box
        boxes = torchvision.ops.box_convert(boxes, "xyxy", "cxcywh")
        boxes[..., [0, 2]] *= (self.rcnn.backbone.body.layer4.v.shape[-1] / img.shape[-1])
        boxes[..., [1, 3]] *= (self.rcnn.backbone.body.layer4.v.shape[-2] / img.shape[-2])
        boxes = torchvision.ops.box_convert(boxes, "cxcywh", "xyxy")
        boxes = torchvision.ops.clip_boxes_to_image(boxes.ceil().to(dtype=torch.int32), img.shape[2:])

        Vs = []
        for i in range(boxes.shape[0]):
            V = []
            for box in boxes[i]:
                V.append(self.rcnn.backbone.body.layer4.v[i, :, box[1]:box[3], box[0]:box[2]].norm(p=2, dim=(1, 2)))
            Vs.append(torch.vstack(V).unsqueeze(0))
        
        return torch.vstack(Vs)


class TD(torch.nn.Module):
    def __init__(self, w_dim=300, v_dim=2048, hid_dim=512, N=3129):
        """
        
        :param w_dim: Word embedding dim.
        :param v_dim: Feature vector dim.
        :param hid_dim: Hidden dim of GRU.
        :param N: Num of candidate answers.
        """
        # Question Embedding
        super(TD, self).__init__()
        glove_vectors= torchtext.vocab.GloVe(name="840B", dim=w_dim)  # ?
        self.emb = torch.nn.Embedding.from_pretrained(glove_vectors.vectors, padding_idx=0, freeze=False)
        self.gru = torch.nn.GRU(input_size=w_dim, hidden_size=hid_dim, num_layers=1, bidirectional=False, batch_first=True)
        # Image Attention
        self.fa = GatedTanh(v_dim + hid_dim, hid_dim)
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
        :param q: Question, shape=[B, 14]
        """
        # Question Embedding
        q = self.emb(q)
        _, hidden = self.gru(q)  # hidden.shape=[1, B, H]
        q = hidden.reshape(hidden.shape[1], 1, -1)  # [B, 1, H]
        # Image Attention
        a = torch.concat((v, q.expand(-1, v.shape[1], -1)), dim=-1)
        a = self.linear1(self.fa(a))  # [B, K, 1]
        a = torch.softmax(a, dim=-2)  # [B, K, 1]
        v_hat = torch.mul(a, v).sum(dim=-2, keepdim=False)  # [B, D]
        # Multimodal Fusion
        h = torch.mul(self.fq(q.squeeze(1)), self.fv(v_hat))  # [B, H]
        # Output Classifier
        return torch.sigmoid(self.linear2(self.fo_text(h)) + self.linear3(self.fo_img(h)))  # [B, N]


class GatedTanh(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GatedTanh, self).__init__()
        self.linear1 = torch.nn.Linear(in_dim, out_dim, bias=True)
        self.linear2 = torch.nn.Linear(in_dim, out_dim, bias=True)
    
    def forward(self, x):
        y_hat = torch.tanh(self.linear1(x))
        g = torch.sigmoid(self.linear2(x))
        y = torch.mul(y_hat, g)
        return y
