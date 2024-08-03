import torch
from torch import nn
from model.Embedding_and_Position import EmbeddingwithPosition
from model.Encoder_Block import EncoderBlock
from config.config import *

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, q_k_size, v_size, f_size, head, nblocks, dropout=0.1, seq_max_len=5000) -> None:
        super().__init__()
        self.embedding = EmbeddingwithPosition(vocab_size=vocab_size, embedding_size=embedding_size)

        self.encoder_blocks = nn.ModuleList()
        for _ in range(nblocks):
            self.encoder_blocks.append(EncoderBlock(embedding_size=embedding_size, q_k_size=q_k_size, v_size=v_size, f_size=f_size, head=head))


    def forward(self,x): #x:(batch_size,seq_len)
        pad_mask = (x==PAD_IDX).unsqueeze(1) #x:(batch_size,1,seq_len)
        pad_mask = pad_mask.expand(x.size()[0],x.size()[1],x.size()[1]) #x:(batch_size,seq_len,seq_len)
        pad_mask = pad_mask.to(DEVICE)
        # print(pad_mask)
        x = self.embedding(x)
        for block in self.encoder_blocks:
            x = block(x, pad_mask)#x:(batch_size,seq_len,embedding_size)
        return x
    


