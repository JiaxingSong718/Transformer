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

## Train

```
python train.py --dataset De2En --weights ./checkpoints/model_De2En.pth --epochs 150 --batch-size 256
```

## Decect

```
python detect.py --dataset De2En --weights ./checkpoints/model_De2En.pth --sentence "Zwei Männer unterhalten sich mit zwei Frauen."
```

## Reference
