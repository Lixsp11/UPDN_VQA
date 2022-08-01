import time
import json
import torch
import config
import torch.utils.data
from model.TopDownAttention_m import TDAttention
from dataset.VQA2 import Dictionary, VQAFeatureDataset
# from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter


@torch.no_grad()
def compute_acc(ann_hat, ann):
    return ann[range(ann.shape[0]), ann_hat.argmax(dim=-1)]

@torch.no_grad()
def train_log(epoch, iteration, loss, time_, anns_hat, anns, writer):
    acc = compute_acc(torch.sigmoid(anns_hat), anns).sum() / anns_hat.shape[0]
    log = f"Epoch={epoch:03}, Iteration={iteration:06}, Loss={loss.detach().item():6.5f}, " \
          f"Acc={acc.item():6.5f}, Time={time.time() - time_:2.4f}"
    print(log)
    with open('train.log', 'a') as f:
        f.write(log + '\n')
    writer.add_scalar('Train CELoss', loss.detach().item(), global_step=iteration)
    writer.add_scalar('Train Acc', acc.item(), global_step=iteration)

@torch.no_grad()
def eval_log(epoch, iteration, loss, yesno_acc, other_acc, number_acc, overall_acc, writer):
    log = f"\nLoss={loss:6.5f}, Yes/No-Acc={yesno_acc:6.5f}, Other-Acc={other_acc:6.5f}, " \
          f"Number-Acc={number_acc:6.5f}, Overall-Acc={overall_acc:6.5f}\n"
    print(log)
    with open('train.log', 'a') as f:
        f.write(log + '\n')
    writer.add_scalar('Eval CELoss', loss, global_step=iteration)
    writer.add_scalar('Eval Yes/No-Acc', yesno_acc, global_step=iteration)
    writer.add_scalar('Eval Other-Acc', other_acc, global_step=iteration)
    writer.add_scalar('Eval Number-Acc', number_acc, global_step=iteration)
    writer.add_scalar('Eval Overall-Acc', overall_acc, global_step=iteration)

@torch.no_grad()
def eval_model(model, val_loader, Loss, qidx2type):
    loss_total, yesno_total, other_total, number_total, overall_total = 0., 0., 0., 0., 0.
    loss_count, yesno_count, other_count, number_count, overall_count = 0., 0., 0., 0., 0.
    for feats, qus, anns, q_idxs in val_loader:
        feats, qus = feats.cuda(non_blocking=True), qus.cuda(non_blocking=True)
        anns = anns.cuda()
        anns_hat = model(feats, qus)
        
        loss_total += (Loss(anns_hat, anns) * anns.shape[-1]).sum().item()
        loss_count += anns_hat.shape[0]
        
        accs = compute_acc(anns_hat, anns)
        
        overall_total += accs.sum().item()
        overall_count += anns_hat.shape[0]
        for acc, q_idx in zip(accs, q_idxs):
            qtype = qidx2type[str(q_idx.item())]
            if qtype == 'yes/no':
                yesno_total += acc.item()
                yesno_count += 1
            elif qtype == 'other':
                other_total += acc.item()
                other_count += 1
            elif qtype == 'number':
                number_total += acc.item()
                number_count += 1
    return loss_total / loss_count, yesno_total / yesno_count, other_total / other_count, \
            number_total / number_count, overall_total / overall_count


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    writer = SummaryWriter("../tf-logs/")
    # Dataset
    with open('util/qid2type_v2.json','r') as f:
        qidx2type=json.load(f)
    dictionary = Dictionary.load_from_file('data/dictionary.pkl')
    print("Building train dataset...")
    train_dataset = VQAFeatureDataset('train', dictionary)
    print("Building test dataset...")
    val_dataset = VQAFeatureDataset('val', dictionary)
    # Datloader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, 
                                               shuffle=True, num_workers=8, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size, 
                                             shuffle=True, num_workers=8, pin_memory=True)
    # Model
    model = TDAttention('data/glove6b_init_300d.npy', hid_dim=config.num_hid, N=config.ann_num_classes).cuda()
    # Loss
    Loss = torch.nn.BCEWithLogitsLoss().cuda()
    # Optim
    optim = torch.optim.Adamax(model.parameters())
    # scaler = GradScaler()
    
    iteration, epoch, time_ = 0, 0, time.time()
    while iteration < 1e6:
        epoch += 1
        model.train()
        for feats, qus, anns, _ in train_loader:
            iteration += 1
            # with autocast():
            feats, qus = feats.cuda(non_blocking=True), qus.cuda(non_blocking=True)
            anns = anns.cuda()
            anns_hat = model(feats, qus)
            loss = Loss(anns_hat, anns)
            loss *= anns.shape[-1]  # ?
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            # scaler.scale(loss).backward()
            # scaler.step(optim)
            # scaler.update()
            optim.step()
            optim.zero_grad()

            if iteration % 400 == 0:
                train_log(epoch, iteration, loss, time_, anns_hat, anns, writer)
                time_ = time.time()

        model.eval()
        eval_acc = eval_model(model, val_loader, Loss, qidx2type) 
        eval_log(epoch, iteration, *eval_acc, writer)
        torch.save(model.state_dict(), 'TD.pt')