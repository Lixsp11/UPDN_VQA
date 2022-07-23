import os
import sys
import csv
import base64
import torch
sys.path.append('..')
from BUTD import config

csv.field_size_limit(sys.maxsize)

if __name__ == "__main__":
    if not os.path.exists(config.trainval_feature):
        os.mkdir(config.trainval_feature)
    save_path = os.path.join(config.trainval_feature, "COCO_trainval_{:012}.pkl")

    fieldnames = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']
    with open(config.feature_tsv, 'r', encoding='ascii') as f:
        reader = csv.DictReader(f, delimiter='\t', fieldnames=fieldnames)
        
        for i, item in enumerate(reader):
            feature = base64.b64decode(item['features'])
            feature = torch.frombuffer(feature, dtype=torch.float32).reshape((36, -1))
            torch.save(feature, save_path.format(int(item['image_id'])))
            if i % 1000 == 0:
                print(f"Preprocess feature:{i}", end='\r')
        print("")
