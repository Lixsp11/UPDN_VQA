# BUTD_vqa
Implemention of the BUTD model proposed by paper "Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering" for VQA task

## Data preprocess

We use the data and preprocess proposed in [CSS-VQA](https://github.com/yanxinzju/CSS-VQA), but only use VQA-V2 data

- Downalod data

  ```bash
  bash tools/download.sh
  ```

  or download form aliyunpan channel https://www.aliyundrive.com/drive/folder/62e68d731ba2996a5a594d04828a3041fde42688

- Process the data

  ```bash
  bash tools/process.sh 
  ```

## Train

- To train original model, modified import statement in `train.py` to

  ```python
  from model.TopDownAttention import TDAttention
  ```


- To train changed model(mainly follows the change of `tpdn` model in [CSS-VQA](https://github.com/yanxinzju/CSS-VQA)), modified import statement in `train.py` to

  ```python
  from model.TopDownAttention_m import TDAttention
  ```

then

  ```python
  python train.py
  ```
