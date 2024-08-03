import torch 

# 设备
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 最长序列（受限于postition emb）
SEQ_MAX_LEN=5000

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

class Tools():
    def __init__(self) -> None:
        pass

    