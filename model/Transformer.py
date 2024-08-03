import torch
from torch import nn
from model.Encoder import Encoder
from model.Decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, encoder_vocab_size, decoder_vocab_size, embedding_size, q_k_size, v_size, f_size, nblocks, head, dropout=0.1, seq_max_len=5000) -> None:
        super().__init__()

        self.encoder = Encoder(vocab_size=encoder_vocab_size, embedding_size=embedding_size,q_k_size=q_k_size,v_size=v_size,f_size=f_size,head=head,nblocks=nblocks)
        self.decoder = Decoder(vocab_size=decoder_vocab_size,embedding_size=embedding_size,q_k_size=q_k_size,v_size=v_size,f_size=f_size,nblocks=nblocks,head=head)

    def forward(self, encoder_x, decoder_x):
        encoder_output = self.encode(encoder_x)
        decoder_output = self.decode(decoder_x, encoder_output, encoder_x)

        return decoder_output
    

    def encode(self, encoder_x):
        encoder_output = self.encoder(encoder_x)
        return encoder_output
    
    def decode(self, decoder_x, encoder_output, encoder_x):
        decoder_output = self.decoder(decoder_x, encoder_output, encoder_x)
        return decoder_output