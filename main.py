import time
import config
import torch
import torchvision
import torch.utils.data
from BUTD import BU, TD, VQA2
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter


torchvision.set_image_backend('accimage')

def cuda_test():
    print(f"torch {torch.__version__}, cuda.is_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda={torch.version.cuda}, cudnn={torch.backends.cudnn.version()}")
        print(torch.cuda.get_device_name())
        print('\n')

@torch.no_grad()
def compute_acc(ann_hat, ann):
    topk_acc = 0.
    for y_hat, y in zip(ann_hat, ann):
        k = (y == 1.).sum().item()
        if k != 0:
            _, indics_hat = torch.topk(y_hat, k=k)
            _, indics = torch.topk(y, k=k)
            topk_acc += len(set(indics_hat.tolist()) & set(indics.tolist())) * 1.0 / k
        else:
            continue
    topk_acc /= ann.shape[0]
    top1_acc = (ann_hat.argmax(dim=-1) == ann.argmax(dim=-1)).sum() / ann_hat.shape[0]
    return topk_acc, top1_acc.item()

@torch.no_grad()
def log_data():
    global epoch, iteration, loss, time_, ann_hat, ann
    topk_acc, top1_acc = compute_acc(ann_hat.cpu(), ann.cpu())

    log = f"Epoch={epoch:03}, Iteration={iteration:06}, Loss={loss.detach().item():6.5f}, " \
          f"Top1 Acc={top1_acc:6.5f}, Topk Acc={topk_acc:6.5f}, " \
          f"Time={time.time() - time_:2.4f}"
    print(log)
    with open('train.log', 'a') as f:
        f.write(log + '\n')
    writer.add_scalar('Train CELoss', loss.detach().item(), global_step=iteration)
    writer.add_scalar('Train Top1-Acc', top1_acc, global_step=iteration)
    writer.add_scalar('Train Topk-Acc', topk_acc, global_step=iteration)
    time_ = time.time()


@torch.no_grad()
def eval_model():
    global model, val_loader, Loss, time_, iteration
    print("##EVAL## ", end="")
    model.eval()
    loss, topk_acc, top1_acc = [], [], []
    for img, qu, ann in val_loader:
        img, ann = img.cuda(non_blocking=True), ann.cuda()
        ann_hat = model(img, qu)
        loss.append(Loss(ann_hat, ann).cpu().item())
        topk_acc_, top1_acc_ = compute_acc(ann_hat.cpu(), ann.cpu())
        topk_acc.append(topk_acc_)
        top1_acc.append(top1_acc_)
    loss = torch.tensor(loss).mean().item()
    topk_acc = torch.tensor(topk_acc).mean().item()
    top1_acc = torch.tensor(top1_acc).mean().item()
    
    log = f"Loss={loss:6.5f}, " \
          f"Top1 Acc={top1_acc:6.5f}, Topk Acc={topk_acc:6.5f}, " \
          f"Time={time.time() - time_:2.4f}"
    print(log)
    with open('train.log', 'a') as f:
        f.write(log + '\n')
    writer.add_scalar('Eval CELoss', loss, global_step=iteration)
    writer.add_scalar('Eval Top1-Acc', top1_acc, global_step=iteration)
    writer.add_scalar('Eval Topk-Acc', topk_acc, global_step=iteration)
    
    model.train()
    time_ = time.time()

if __name__ == "__main__":
    cuda_test()

    writer = SummaryWriter("tf-logs/")
    train_dataset = VQA2(config.trainval_feature, config.train_questions, config.train_annotations, 
                            config.ann_num_classes, mode='feature')
    val_dataset = VQA2(config.trainval_feature, config.val_questions, config.val_annotations, 
                            config.ann_num_classes, mode='feature')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, 
                                                num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True, 
                                                num_workers=4, pin_memory=True)
    
    model = TD(N=config.ann_num_classes).cuda()
    Loss = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adadelta(model.parameters(), lr=config.lr)
    # scaler = GradScaler()

    iteration, epoch, time_ = 0, 0, time.time()
    while iteration < 1e5:
        epoch += 1
        for img, qu, ann in train_loader:
            img, ann = img.cuda(non_blocking=True), ann.cuda()
            # with autocast():
            ann_hat = model(img, qu)
            loss = Loss(ann_hat, ann)
            loss.backward()
            # scaler.scale(loss).backward()
            # scaler.step(optim)
            # scaler.update()
            optim.step()
            optim.zero_grad()

            if iteration % 100 == 0:
                log_data()
            if iteration % 3000 == 0:
                eval_model()
                torch.save(model.state_dict(), 'TD.pt')
            iteration += 1
