import torch
import torchvision


class BUAttention(torch.nn.Module):
    def __init__(self, K=36):
        """

        :param K: Num of features.
        """
        super(BUAttention, self).__init__()
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