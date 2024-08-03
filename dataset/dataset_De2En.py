from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import multi30k, Multi30k
from torch.utils.data import Dataset
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

# 加载数据集
# multi30k.URL["train"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
# multi30k.URL["valid"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"
# train_dataset = list(Multi30k(split='train', language_pair=('de', 'en')))
# print(train_dataset[3])

class De2EnDataset(Dataset):
    def __init__(self, data_file, dataset_type) -> None:
        super().__init__()
        self.dataset_type = dataset_type
        self.dataset = []
        if self.dataset_type == 'train':
            with open(data_file+'train.de', 'r', encoding='utf-8') as file:
                dataset_De = file.read()
            dataset_De = dataset_De.splitlines()
            with open(data_file+'train.en', 'r', encoding='utf-8') as file:
                dataset_En = file.read()
            dataset_En = dataset_En.splitlines()
            self.dataset = list(zip(dataset_De, dataset_En))
        else:
            with open(data_file+'val.de', 'r', encoding='utf-8') as file:
                dataset_De = file.read()
            dataset_De = dataset_De.splitlines()
            with open(data_file+'val.en', 'r', encoding='utf-8') as file:
                dataset_En = file.read()
            dataset_En = dataset_En.splitlines()
            self.dataset = list(zip(dataset_De, dataset_En))

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return self.dataset[index]

train_dataset = De2EnDataset(data_file='./dataset/dataset_De2En/train/',dataset_type='train')
val_dataset = De2EnDataset(data_file='./dataset/dataset_De2En/val/',dataset_type='val')
# print(train_dataset[3])

# 创建分词器
de_tokenizer = get_tokenizer('spacy', language='de_core_news_sm')
en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

#生成词表
# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

de_tokens = [] # 德语token列表
en_tokens = [] # 英语token列表
for de,en in tqdm(train_dataset, desc="Processing"):
    de_tokens.append(de_tokenizer(de))
    en_tokens.append(en_tokenizer(en))

# print(en_tokens[:10])

de_vocab = build_vocab_from_iterator(de_tokens, specials=special_symbols, special_first=True) # 德语token词表
de_vocab.set_default_index(UNK_IDX)
en_vocab = build_vocab_from_iterator(en_tokens, specials=special_symbols, special_first=True) # 英语token词表
en_vocab.set_default_index(UNK_IDX)
# print(en_vocab['<pad>'])

# 保存词表到文件
with open('./dataset/dataset_De2En/en_vocab.pkl', 'wb') as f:
    pickle.dump(en_vocab, f)

with open('./dataset/dataset_De2En/de_vocab.pkl', 'wb') as f:
    pickle.dump(de_vocab, f)

with open('./dataset/dataset_De2En/en_vocab.pkl', 'rb') as f:
    en_vocab = pickle.load(f)

with open('./dataset/dataset_De2En/de_vocab.pkl', 'rb') as f:
    de_vocab = pickle.load(f)

# 句子特征预处理
def de_preprocess(de_sentence):
    tokens = de_tokenizer(de_sentence)
    tokens = [special_symbols[2]] + tokens + [special_symbols[3]]
    ids = de_vocab(tokens)
    return tokens,ids

def en_preprocess(en_sentence):
    tokens = en_tokenizer(en_sentence)
    tokens = [special_symbols[2]] + tokens + [special_symbols[3]]
    ids = en_vocab(tokens)
    return tokens,ids

def prepare_dataset(dataset):

    encoder_x = []
    decoder_x = []

    for de, en in tqdm(dataset, desc="Processing"):
        de_tokens, de_ids = de_preprocess(de)
        en_tokens, en_ids = en_preprocess(en)

        if len(de_ids) > SEQ_MAX_LEN:
            de_ids = de_ids[:SEQ_MAX_LEN]
        if len(en_ids) > SEQ_MAX_LEN:
            en_ids = en_ids[:SEQ_MAX_LEN]
        encoder_x.append(de_ids)
        decoder_x.append(en_ids)
    return encoder_x, decoder_x

# train_encoder_x, train_decoder_x = prepare_dataset(train_dataset)
# torch.save([train_encoder_x, train_decoder_x], './dataset/dataset_De2En/train.pt')
# val_encoder_x, val_decoder_x = prepare_dataset(val_dataset)
# torch.save([val_encoder_x, val_decoder_x], './dataset/dataset_De2En/val.pt')
    

if __name__ == '__main__':
    # 词表大小
    print('de vocab:', len(de_vocab))
    print('en vocab:', len(en_vocab))

    # 特征预处理
    de_sentence,en_sentence=train_dataset[5]
    print('de preprocess:',*de_preprocess(de_sentence))
    print('en preprocess:',*en_preprocess(en_sentence))