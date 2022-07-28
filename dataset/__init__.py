import torch

def collate_fn(batch):
    feats, querys, anns = [], [], []
    for feat, query, ann in batch:
        feats.append(feat)
        querys.append(query)
        anns.append(ann)
    feats = torch.stack(feats)
    querys = torch.nn.utils.rnn.pad_sequence(querys, batch_first=True, padding_value=0)
    anns = torch.stack(anns)
    return feats, querys, anns