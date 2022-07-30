import time
import torch

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
def train_log(epoch, iteration, loss, time_, anns_hat, anns, writer):
    topk_acc, top1_acc = compute_acc(anns_hat.cpu(), anns.cpu())

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
def eval_log(epoch, iteration, eval_acc, writer):
    log = f"\nTop1-Acc={eval_acc[0]:6.5f}, Topk-Acc={eval_acc[1]:6.5f}\n"
    print(log)
    with open('train.log', 'a') as f:
        f.write(log + '\n')
    writer.add_scalar('Eval Top1-Acc', eval_acc[0], global_step=iteration)
    writer.add_scalar('Eval Topk-Acc', eval_acc[1], global_step=iteration)

@torch.no_grad()
def eval_model(model, val_loader):
    topk_acc, top1_acc = [], []
    for feats, qus, anns in val_loader:
        feats, anns = feats.cuda(non_blocking=True), anns.to(device=device)
        anns_hat = model(feats, qus)
        topk_acc_, top1_acc_ = compute_acc(anns_hat.cpu(), anns.cpu())
        topk_acc.append(topk_acc_)
        top1_acc.append(top1_acc_)
    loss = torch.tensor(loss).mean().item()
    topk_acc = torch.tensor(topk_acc).mean().item()
    top1_acc = torch.tensor(top1_acc).mean().item()
    return top1_acc, topk_acc