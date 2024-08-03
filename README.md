# Transformer(Pytorch) - Machine Translation

使用Pytorch复现Transformer，并完成机器翻译任务(法语 -> 英文、英文 -> 中文)


## Install

```
git clone https://github.com/JiaxingSong718/Transformer.git  # clone
cd Transformer
```

## Environment

```
conda create -n Transformer python=3.7
conda activate Transformer
pip install -r requirements.txt  # install
```
## Dataset
法语 -> 英文数据集：
数据集[train](https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz) [valid](https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz)
英文 -> 中文数据集：
[repo](https://github.com/brightmart/nlp_chinese_corpus?tab=readme-ov-file)中的翻译语料(translation2019zh)
note:由于设备原因，只取了translation2019zh中train的前58000个,valid的前2028个。
## Train

```
python train.py --dataset De2En --weights ./checkpoints/model_De2En.pth --epochs 150 --batch-size 256
```

## Decect

```
python detect.py --dataset De2En --weights ./checkpoints/model_De2En.pth --sentence "Zwei Männer unterhalten sich mit zwei Frauen."
```

## Reference
