import torch
import orjson
import torchvision
from BUTD import BU, TD
from types import SimpleNamespace
from torchvision.io.image import read_image

torchvision.set_image_backend('accimage')
config = SimpleNamespace(**orjson.loads(open('config.json', "rb").read()))

def cuda_test():
    print(f"torch {torch.__version__}, cuda.is_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda={torch.version.cuda}, cudnn={torch.backends.cudnn.version()}")
        print(torch.cuda.get_device_name())
        print('\n')

if __name__ == "__main__":
    cuda_test()

    bu_model = BU().cuda()
    td_model = TD().cuda()
    bu_model.eval()


    img = torchvision.io.read_image("grace_hopper_517x606.jpg").unsqueeze(0).cuda()
    v = bu_model(torch.vstack((img, img, img)))
    y = td_model(v, torch.randint(0, 10, (3, 14), device='cuda'))
    print(y.shape)
