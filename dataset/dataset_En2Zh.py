from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset
import json
import pickle
import sys
import os
import torch
from tqdm import tqdm

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目的根目录
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
from config.config import SEQ_MAX_LEN

class En2ZhDataset(Dataset):
    def __init__(self, data_file) -> None:
        super().__init__()
        self.dataset = []
        with open(data_file, 'r', encoding='utf-8') as file:
            dataset = json.load(file)

        self.dataset = [(item['english'], item['chinese']) for item in dataset]

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return self.dataset[index]

train_dataset = En2ZhDataset(data_file='./dataset/dataset_En2Zh/translation2019zh_train.json')
train_dataset = train_dataset[:58000]
val_dataset = En2ZhDataset(data_file='./dataset/dataset_En2Zh/translation2019zh_valid.json')
val_dataset = val_dataset[:2028]
# print(train_dataset[2])

# 创建分词器
en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
zh_tokenizer = get_tokenizer('spacy', language='zh_core_web_sm')

# tokens = zh_tokenizer('微风推着我去爱抚它的长耳朵')
# print(tokens)

#生成词表
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
zh_tokens = [] # 中文token列表
en_tokens = [] # 英语token列表
for en, zh in tqdm(train_dataset, desc="Processing"):
    en_tokens.append(en_tokenizer(en))
    zh_tokens.append(zh_tokenizer(zh))

# print(en_tokens[:10])

en_vocab = build_vocab_from_iterator(en_tokens, specials=special_symbols, special_first=True) # 英语token词表
en_vocab.set_default_index(UNK_IDX)
zh_vocab = build_vocab_from_iterator(zh_tokens, specials=special_symbols, special_first=True) # 中文token词表
zh_vocab.set_default_index(UNK_IDX)
# print(en_vocab['<pad>'])
# 保存词表到文件
with open('en_vocab.pkl', 'wb') as f:
    pickle.dump(en_vocab, f)

with open('zh_vocab.pkl', 'wb') as f:
    pickle.dump(zh_vocab, f)

# 从文件读取词表
with open('./dataset/dataset_En2Zh/en_vocab.pkl', 'rb') as f:
    en_vocab = pickle.load(f)

with open('./dataset/dataset_En2Zh/zh_vocab.pkl', 'rb') as f:
    zh_vocab = pickle.load(f)

# 句子特征预处理
def en_preprocess(en_sentence):
    tokens = en_tokenizer(en_sentence)
    tokens = [special_symbols[2]] + tokens + [special_symbols[3]]
    ids = en_vocab(tokens)
    return tokens,ids

def zh_preprocess(zh_sentence):
    tokens = zh_tokenizer(zh_sentence)
    tokens = [special_symbols[2]] + tokens + [special_symbols[3]]
    ids = zh_vocab(tokens)
    return tokens,ids

def prepare_dataset(dataset):

    encoder_x = []
    decoder_x = []

    for en, zh in tqdm(dataset, desc="Processing"):
        zh_tokens, zh_ids = zh_preprocess(zh)
        en_tokens, en_ids = en_preprocess(en)

        if len(zh_ids) > SEQ_MAX_LEN:
            zh_ids = zh_ids[:SEQ_MAX_LEN]
        if len(en_ids) > SEQ_MAX_LEN:
            en_ids = en_ids[:SEQ_MAX_LEN]
        encoder_x.append(en_ids)
        decoder_x.append(zh_ids)
    return encoder_x, decoder_x

# train_encoder_x, train_decoder_x = prepare_dataset(train_dataset)
# torch.save([train_encoder_x, train_decoder_x], './dataset/dataset_De2En/train.pt')
# val_encoder_x, val_decoder_x = prepare_dataset(val_dataset)
# torch.save([val_encoder_x, val_decoder_x], './dataset/dataset_De2En/val.pt')

if __name__ == '__main__':
    # 词表大小
    print('de vocab:', len(zh_vocab))
    print('en vocab:', len(en_vocab))

    # 特征预处理
    en_sentence,zh_sentence=train_dataset[5]
    print('de preprocess:',*zh_preprocess(zh_sentence))
    print('en preprocess:',*en_preprocess(en_sentence))