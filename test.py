import torch
import torchvision
from BUTD import BU, TD

if __name__ == "__main__":
    bu_model = BU().cuda()
    bu_model.eval()
    td_model = TD()

    img = torchvision.io.read_image("grace_hopper_517x606.jpg").unsqueeze(0).cuda()
    v = bu_model(torch.vstack((img, img, img)))
    y = td_model(v, torch.randint(0, 10, (3, 14), device='cuda'))
    print(y.shape)
