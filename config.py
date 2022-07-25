root_path = "/root/VQA2.0/"

train_questions = root_path + "v2_OpenEnded_mscoco_train2014_questions.json"
train_annotations = root_path + "train_target.json"

val_questions = root_path + "v2_OpenEnded_mscoco_val2014_questions.json"
val_annotations = root_path + "val_target.json"

feature_tsv = root_path + "trainval_resnet101_faster_rcnn_genome_36.tsv"
trainval_feature = root_path + "trainval_resnet101_faster_rcnn_genome_36"


ann_num_classes = 4902

batch_size = 768
lr = 1.0 
