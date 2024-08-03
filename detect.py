import argparse
import torch
from dataset.dataset_De2En import de_preprocess as de_preprocess_de, en_vocab as en_vocab_de
from dataset.dataset_En2Zh import zh_preprocess, en_vocab as en_vocab_zh
from config.config import *
import sys
import os

# 动态添加包含 Transformer 模块的目录到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, 'model')
sys.path.append(model_dir)

def translate(transformer, sentence, lang):
    if lang == 'De2En':
        tokens, ids = de_preprocess_de(sentence)
        en_vocab = en_vocab_de
    elif lang == 'Zh2En':
        tokens, ids = zh_preprocess(sentence)
        en_vocab = en_vocab_zh
    else:
        raise ValueError(f"Unsupported language pair: {lang}")
    print(f"Tokens: {tokens}")
    print(f"IDs: {ids}")
    if len(ids) > SEQ_MAX_LEN:
        ids = ids[:SEQ_MAX_LEN]

    # Encoder
    encoder_x_batch = torch.tensor([ids], dtype=torch.long).to(DEVICE)
    encoder_z = transformer.encode(encoder_x_batch)

    # Decoder
    en_token_ids = [BOS_IDX]
    while len(en_token_ids) < SEQ_MAX_LEN:
        decoder_x_batch = torch.tensor([en_token_ids],dtype=torch.long).to(DEVICE) #准备decoder输入
        decoder_z = transformer.decode(decoder_x_batch,encoder_z,encoder_x_batch) #decoder解码输出
        next_token_probs = decoder_z[0,decoder_z.size(1)-1,:]   # 序列下一个词的概率
        next_token_id = torch.argmax(next_token_probs) # 下一个词ID
        en_token_ids.append(next_token_id)


        if next_token_id == EOS_IDX:
            break
    print(f"Token IDs: {en_token_ids}")
    # 生成翻译结果
    en_token_ids = [id for id in en_token_ids if id not in [BOS_IDX, EOS_IDX, PAD_IDX, UNK_IDX]]
    en_tokens = en_vocab.lookup_tokens(en_token_ids)
    return ' '.join(en_tokens)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['De2En', 'Zh2En'], default='De2En', help='Specify the dataset to use')
    parser.add_argument('--weights', type=str, default='checkpoints/model_De2En.pth', help='Path to the model weights')
    parser.add_argument('--sentence', type=str, default='Zwei Männer unterhalten sich mit zwei Frauen.', help='Sentence to translate')
    args = parser.parse_args()

    transformer = torch.load(args.weights)
    transformer.eval()

    translation = translate(transformer, args.sentence, args.dataset)
    print(translation)
