import os
import nltk
import csv
import numpy as np
import matplotlib.pyplot as plt
from config.config import *
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

# Ensure nltk resources are downloaded
# nltk.download('punkt', download_dir='./config/nltk_data')
# nltk.data.path.append('./config/nltk_data')

def calculate_bleu_score(references, hypotheses):
    smoothie = SmoothingFunction().method1
    return corpus_bleu(references, hypotheses, smoothing_function=smoothie) * 100  # BLEU score is usually reported as a percentage

def calculate_ppl_score(hypotheses, dataloader,vocab_size):
    # PPL calculation assumes a vocab size and proper normalization
    vocab_size = vocab_size  # or en_vocab based on your language model
    total_log_prob = 0
    total_words = 0
    for i, (pad_encoder_x, pad_decoder_x) in enumerate(dataloader):
        pad_decoder_x = pad_decoder_x[:, :-1].to(DEVICE)  # Remove <end> token
        if len(hypotheses) == 0:
            break
        for j in range(pad_decoder_x.size(0)):
            target = pad_decoder_x[j].cpu().tolist()
            pred = hypotheses[i * pad_decoder_x.size(0) + j]
            if target and pred:
                log_prob = -np.log2(np.mean([pred.count(w) / len(pred) for w in target if w in pred]) + 1e-9)  # Adding epsilon for stability
                total_log_prob += log_prob
                total_words += len(target)

    if total_words > 0:
        ppl_score = np.exp(total_log_prob / total_words)
    else:
        ppl_score = float('inf')
    
    return ppl_score


def save_results_to_csv(train_losses, val_losses, bleu_scores, ppl_scores, epoches, save_dir):
    csv_file = os.path.join(save_dir, 'training_results.csv')
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Loss', 'Validation Loss', 'BLEU Score', 'PPL Score'])
        for epoch in range(epoches):
            writer.writerow([epoch + 1, train_losses[epoch], val_losses[epoch], bleu_scores[epoch], ppl_scores[epoch]])

def plot_losses(train_losses, val_losses, bleu_scores, ppl_scores, epoches, save_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(range(epoches), train_losses, label='Train Loss', color='blue', linestyle='-', marker='o')
    plt.plot(range(epoches), val_losses, label='Validation Loss', color='red', linestyle='-', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'losses.png'))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(range(epoches), bleu_scores, label='BLEU Score', color='green', linestyle='-', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('BLEU Score')
    plt.title('BLEU Score over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'bleu_score.png'))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(range(epoches), ppl_scores, label='PPL Score', color='purple', linestyle='-', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('PPL Score')
    plt.title('PPL Score over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'ppl_score.png'))
    plt.close()