import sys
import csv
import torch
import base64
import config
import numpy as np
csv.field_size_limit(sys.maxsize)

if __name__ == "__main__":
    fieldnames = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']
    features = dict()
    with open(config.bottom_up_feature, 'r', encoding='ascii') as f:
        reader = csv.DictReader(f, delimiter='\t', fieldnames=fieldnames)
        
        for item in reader:
            feature = base64.b64decode(item['features'])
            feature = torch.frombuffer(feature, dtype=torch.float32).reshape((36, -1))
            features[item['image_id']] = feature
            break
    print(features['150367'].shape)

    f = pd.read_csv(config.bottom_up_feature, index_col='image_id', dtype=np.float32, chunksize=5000)
    print(f['150367'])