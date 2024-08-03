from torch import nn
import torch
from model.Embedding_and_Position import EmbeddingwithPosition
from model.Decoder_Block import DecoderBlock
from config.config import DEVICE, PAD_IDX

class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, q_k_size, v_size, f_size, nblocks, head, dropout=0.1, seq_max_len=5000) -> None:
        super().__init__()
        self.embedding = EmbeddingwithPosition(vocab_size=vocab_size,embedding_size=embedding_size,dropout=dropout,seq_max_len=seq_max_len)

        self.decoder = nn.ModuleList()
        for _ in range(nblocks):
            self.decoder.append(DecoderBlock(embedding_size=embedding_size,q_k_size=q_k_size,v_size=v_size,f_size=f_size,head=head))

        self.linear = nn.Linear(embedding_size, vocab_size)


    def forward(self,x,encoder_z,encoder_x):
        First_attention_mask = (x==PAD_IDX).unsqueeze(1).expand(x.size()[0],x.size()[1],x.size()[1]).to(DEVICE)
        First_attention_mask = First_attention_mask | torch.triu(torch.ones(x.size()[1],x.size()[1]),diagonal=1).bool().unsqueeze(0).expand(x.size()[0],-1,-1).to(DEVICE)

        Second_attention_mask = (encoder_x==PAD_IDX).unsqueeze(1).expand(encoder_x.size()[0], x.size()[1], encoder_x.size()[1]).to(DEVICE) #(batch_size, target_len, src_len)

        x = self.embedding(x)
        for block in self.decoder:
            x = block(x, encoder_z, First_attention_mask, Second_attention_mask)
        output = self.linear(x) #(batch_size, target_len, vocab_size)
        return output
    
