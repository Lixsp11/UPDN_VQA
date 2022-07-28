import time
import torch
import config
import torch.utils.data
from dataset import collate_fn
from dataset.utils import VQA2feats
from dataset.VQA2 import VQA2Dataset
from model.TopDownAttention import TDAttention
from eval import train_log, eval_log, eval_model
# from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter


if __name__ == "__main__":
    writer = SummaryWriter("../tf-logs/")
    # Dataset
    vqa2feats = VQA2feats(config.trainval_feature)
    train_dataset = VQA2Dataset(vqa2feats, config.train_questions, config.train_annotations, 
                                config.word_dim, config.ann_num_classes)
    val_dataset = VQA2Dataset(vqa2feats, config.val_questions, config.val_annotations, 
                              config.word_dim, config.ann_num_classes)
    # Datloader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, 
                                                num_workers=8, collate_fn=collate_fn, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True, 
                                                num_workers=8, collate_fn=collate_fn, pin_memory=True)
    # Model
    model = TDAttention(N=config.ann_num_classes).cuda()
    # Loss
    Loss = torch.nn.CrossEntropyLoss().cuda()
    # Optim
    optim = torch.optim.Adadelta(model.parameters(), lr=config.lr)
    # scaler = GradScaler()

    iteration, epoch, time_ = 0, 0, time.time()
    while iteration < 1e6:
        epoch += 1
        model.train()
        for feats, qus, anns in train_loader:
            iteration += 1
            # with autocast():
            feats, anns = feats.cuda(non_blocking=True), anns.cuda()
            anns_hat = model(feats, qus)
            loss = Loss(anns_hat, anns)
            
            if iteration % 100 == 0:
                train_log(epoch, iteration, loss, time_, anns_hat, anns, writer)
                time_ = time.time()
            loss.backward()
            # scaler.scale(loss).backward()
            # scaler.step(optim)
            # scaler.update()
            optim.step()
            optim.zero_grad()
        
        model.eval()            
        eval_acc = eval_model(model, val_loader)
        eval_log(epoch, iteration, eval_acc, writer)
        torch.save(model.state_dict(), 'TD.pt')
