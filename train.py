import argparse
from torch import nn
import torch
from model.Transformer import Transformer
from torch.utils.data import Dataset, DataLoader
from config.config import *
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from config.Tools import plot_losses, calculate_ppl_score, calculate_bleu_score,save_results_to_csv
import os
import pickle

class TranslationDataset(Dataset):
    def __init__(self, file) -> None:
        super().__init__()

        self.encoder_x = torch.load(file)[0]
        self.decoder_x = torch.load(file)[1]

    def __len__(self):
        return len(self.encoder_x)
    
    def __getitem__(self, index):
        return self.encoder_x[index], self.decoder_x[index]

def collate_fn(batch):
    encoder_x_batch = []
    decoder_x_batch = []

    for enc_x, dec_x in batch:
        encoder_x_batch.append(torch.tensor(enc_x, dtype=torch.long))
        decoder_x_batch.append(torch.tensor(dec_x, dtype=torch.long))

    pad_encoder_x = pad_sequence(encoder_x_batch, True, PAD_IDX)
    pad_decoder_x = pad_sequence(decoder_x_batch, True, PAD_IDX) 
    return pad_encoder_x, pad_decoder_x

def train(model, dataloader, loss_fn, optimizer):
    model.train()
    total_loss = 0
    for pad_encoder_x, pad_decoder_x in tqdm(dataloader, desc="Training", leave=True, unit="batch"):
        real_decoder_z = pad_decoder_x[:, 1:].to(DEVICE)  # 标签需要去掉第一个词<start>
        pad_encoder_x = pad_encoder_x.to(DEVICE)
        pad_decoder_x = pad_decoder_x[:, :-1].to(DEVICE)  # decoder输入需要去掉最后一个词
        decoder_z = model(pad_encoder_x, pad_decoder_x)

        loss = loss_fn(decoder_z.view(-1, decoder_z.size()[-1]), real_decoder_z.view(-1))  # 把batch中所有的词拉平
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return total_loss / len(dataloader)

def validate(model, dataloader, loss_fn, PPL_vocab):
    model.eval()
    total_loss = 0
    references = []
    hypotheses = []
    
    with torch.no_grad():
        for pad_encoder_x, pad_decoder_x in tqdm(dataloader, desc="Validating", leave=True, unit="batch"):
            real_decoder_z = pad_decoder_x[:, 1:].to(DEVICE)
            pad_encoder_x = pad_encoder_x.to(DEVICE)
            pad_decoder_x = pad_decoder_x[:, :-1].to(DEVICE)
            decoder_z = model(pad_encoder_x, pad_decoder_x)
            loss = loss_fn(decoder_z.view(-1, decoder_z.size()[-1]), real_decoder_z.view(-1))
            total_loss += loss.item()

            # Collect predictions and references for BLEU and PPL calculations
            pred_ids = torch.argmax(decoder_z, dim=-1)
            for i in range(pred_ids.size(0)):
                pred_tokens = pred_ids[i].cpu().tolist()
                ref_tokens = pad_decoder_x[i].cpu().tolist()
                pred_tokens = [t for t in pred_tokens if t != PAD_IDX]
                ref_tokens = [t for t in ref_tokens if t != PAD_IDX]
                if pred_tokens and ref_tokens:
                    hypotheses.append(pred_tokens)
                    references.append([ref_tokens])
    
    bleu_score = calculate_bleu_score(references, hypotheses)
    ppl_score = calculate_ppl_score(hypotheses, dataloader,len(PPL_vocab))
    return total_loss / len(dataloader), bleu_score, ppl_score

def main(args):
    # 创建保存训练结果的目录
    run_id = 1
    while os.path.exists(f'run/train{run_id}'):
        run_id += 1
    save_dir = f'run/train{run_id}'
    os.makedirs(save_dir)

    # 数据集选择
    if args.dataset == 'De2En':
        train_dataset = TranslationDataset('./dataset/dataset_De2En/train.pt')
        val_dataset = TranslationDataset('./dataset/dataset_De2En/val.pt')
        with open('./dataset/dataset_De2En/en_vocab.pkl', 'rb') as f:
            en_vocab_de = pickle.load(f)
        with open('./dataset/dataset_De2En/de_vocab.pkl', 'rb') as f:
            de_vocab = pickle.load(f)
        PPL_vocab = en_vocab_de
    elif args.dataset == 'En2Zh':
        train_dataset = TranslationDataset('./dataset/dataset_En2Zh/train.pt')
        val_dataset = TranslationDataset('./dataset/dataset_De2En/val.pt')
        with open('./dataset/dataset_En2Zh/en_vocab.pkl', 'rb') as f:
            en_vocab_zh = pickle.load(f)
        with open('./dataset/dataset_En2Zh/zh_vocab.pkl', 'rb') as f:
            zh_vocab = pickle.load(f)
        PPL_vocab = zh_vocab
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, persistent_workers=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, persistent_workers=True, collate_fn=collate_fn)

    # 模型
    try:
        transformer = torch.load(args.weights)
    except:
        transformer = Transformer(encoder_vocab_size=len(en_vocab_zh if args.dataset == 'En2Zh' else de_vocab), decoder_vocab_size=len(zh_vocab if args.dataset == 'En2Zh' else en_vocab_de), embedding_size=512, q_k_size=64, v_size=64, f_size=2048, nblocks=6, head=8, dropout=0.1, seq_max_len=SEQ_MAX_LEN).to(DEVICE)
    
    # 损失函数和优化器
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.SGD(transformer.parameters(), lr=1e-3, momentum=0.99)

    # 训练模型
    train_losses = []
    val_losses = []
    bleu_scores = []
    ppl_scores = []
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch + 1}/{args.epochs}')
        avg_train_loss = train(transformer, train_dataloader, loss_fn, optimizer)
        train_losses.append(avg_train_loss)

        avg_val_loss, bleu_score, ppl_score = validate(transformer, val_dataloader, loss_fn, PPL_vocab)
        val_losses.append(avg_val_loss)
        bleu_scores.append(bleu_score)
        ppl_scores.append(ppl_score)

        print(f'Train Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f}')
        print(f'BLEU Score: {bleu_score:.2f} | PPL Score: {ppl_score:.2f}')
        torch.save(transformer, args.weights)

    # 保存训练结果
    save_results_to_csv(train_losses, val_losses, bleu_scores, ppl_scores, args.epochs, save_dir)
    plot_losses(train_losses, val_losses, bleu_scores, ppl_scores, args.epochs, save_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['De2En', 'En2Zh'], default='De2En', help='Specify the dataset to use')
    parser.add_argument('--weights', type=str, default='checkpoints/model_De2En1.pth', help='Initial weights path')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=256, help='Total batch size for all GPUs')
    args = parser.parse_args()

    main(args)
