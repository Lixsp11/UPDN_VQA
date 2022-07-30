import time
import torch
import config
import torch.utils.data
from dataset import collate_fn
from dataset.utils import VQA2feats
from dataset.VQA2 import VQA2Dataset
from model.TopDownAttention import TDAttention
# from eval import train_log, eval_log, eval_model
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter


@torch.no_grad()
def compute_acc(ann_hat, ann):
    anns_hat = torch.sigmoid(anns_hat)
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
def train_log(epoch, iteration, loss, time_, anns_hat, anns, writer):
    topk_acc, top1_acc = compute_acc(anns_hat, anns)
    log = f"Epoch={epoch:03}, Iteration={iteration:06}, Loss={loss.detach().item():6.5f}, " \
          f"Top1-Acc={top1_acc:6.5f}, Topk-Acc={topk_acc:6.5f}, " \
          f"Time={time.time() - time_:2.4f}"
    print(log)
    with open('train.log', 'a') as f:
        f.write(log + '\n')
    writer.add_scalar('Train CELoss', loss.detach().item(), global_step=iteration)
    writer.add_scalar('Train Top1-Acc', top1_acc, global_step=iteration)
    writer.add_scalar('Train Topk-Acc', topk_acc, global_step=iteration)

@torch.no_grad()
def eval_log(epoch, iteration, loss, top1_acc, topk_acc, writer):
    log = f"\nLoss={loss:6.5f}, Top1-Acc={top1_acc:6.5f}, Topk-Acc={topk_acc:6.5f}\n"
    print(log)
    with open('train.log', 'a') as f:
        f.write(log + '\n')
    writer.add_scalar('Eval CELoss', loss, global_step=iteration)
    writer.add_scalar('Eval Top1-Acc', top1_acc, global_step=iteration)
    writer.add_scalar('Eval Topk-Acc', topk_acc, global_step=iteration)

@torch.no_grad()
def eval_model(model, val_loader, Loss):
    loss, topk_acc, top1_acc = [], [], []
    for feats, qus, anns in val_loader:
        feats, qus = feats.cuda(non_blocking=True), qus.cuda(non_blocking=True)
        anns = anns.cuda()
        anns_hat = model(feats, qus)
        loss.append(Loss(anns_hat, anns))
        topk_acc_, top1_acc_ = compute_acc(anns_hat, anns)
        topk_acc.append(topk_acc_)
        top1_acc.append(top1_acc_)
    
    loss = torch.tensor(loss).mean().item()
    topk_acc = torch.tensor(topk_acc).mean().item()
    top1_acc = torch.tensor(top1_acc).mean().item()
    return loss, top1_acc, topk_acc


if __name__ == "__main__":
    writer = SummaryWriter("../tf-logs/")
    # Dataset
    vqa2feats = dict()
    VQA2feats(config.feature_tsv, vqa2feats)
    train_dataset = VQA2Dataset(vqa2feats, config.train_questions, config.train_annotations, 
                            config.word_dim, config.ann_num_classes)
    val_dataset = VQA2Dataset(vqa2feats, config.val_questions, config.val_annotations, 
                            config.word_dim, config.ann_num_classes)
    # Datloader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                                               num_workers=4, collate_fn=collate_fn, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True,
                                             num_workers=4, collate_fn=collate_fn, pin_memory=True)
    # Model
    model = TDAttention(N=config.ann_num_classes).cuda()
    # Loss
    Loss = torch.nn.CrossEntropyLoss().cuda()
    # Optim
    optim = torch.optim.Adadelta(model.parameters(), lr=config.lr)
    scaler = GradScaler()
    
    iteration, epoch, time_ = 0, 0, time.time()
    while iteration < 1e6:
        epoch += 1
        model.train()
        for feats, qus, anns in train_loader:
            iteration += 1
            with autocast():
                feats, qus = feats.cuda(non_blocking=True), qus.cuda(non_blocking=True)
                anns = anns.cuda()
                anns_hat = model(feats, qus)
                loss = Loss(anns_hat, anns)
            # loss.backward()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            # optim.step()
            optim.zero_grad()

            if iteration % 400 == 0:
                train_log(epoch, iteration, loss, time_, anns_hat, anns, writer)
                time_ = time.time()

        if epoch % 10 == 0:
            model.eval()
            eval_acc = eval_model(model, val_loader, Loss) 
            eval_log(epoch, iteration, *eval_acc, writer)
            torch.save(model.state_dict(), 'TD.pt')