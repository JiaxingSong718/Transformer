from torch import nn
import torch
from model.MultiHeadAttention import MultiHeadAttention

class DecoderBlock(nn.Module):
    def __init__(self, embedding_size, q_k_size, v_size, f_size, head) -> None:
        super().__init__()

        # 第一个多头注意力 --> 自注意力
        self.FirstMultiheadAttention = MultiHeadAttention(embedding_size=embedding_size, q_k_size=q_k_size, v_size=v_size, head=head)
        self.z_linear1 = nn.Linear(head*v_size, embedding_size)
        self.addnorm1 = nn.LayerNorm(embedding_size)

        # 第二个多头注意力 --> 多头注意力
        self.SecondMultiheadAttention = MultiHeadAttention(embedding_size=embedding_size, q_k_size=q_k_size, v_size=v_size, head=head)
        self.z_linear2 = nn.Linear(head*v_size, embedding_size)
        self.addnorm2 = nn.LayerNorm(embedding_size)

        # Feedforward结构
        self.feedforward = nn.Sequential(
            nn.Linear(embedding_size, f_size),
            nn.ReLU(),
            nn.Linear(f_size, embedding_size)
        )
        self.addnorm3 = nn.LayerNorm(embedding_size)
    
    def forward(self, x, encoder_z, First_attention_mask, Secon_attention_mask): #x:(batch_size, seq_max_len, embedding_size)
        # 第一个多头
        z = self.FirstMultiheadAttention(x, x, First_attention_mask) #z:(batch_size, seq_max_len, head*v_size)   First_attention_mask的作用在于：1）遮盖decoder序列的pad 2)遮盖decoder Q到每个词后面的词
        z = self.z_linear1(z) #z:(batch_size, seq_max_len, embedding_size) 
        output1 = self.addnorm1(z + x) #output1:(batch_size, seq_max_len, embedding_size)

        # 第二个多头
        z = self.SecondMultiheadAttention(x_q=output1, x_k_v=encoder_z, attention_mask=Secon_attention_mask) #z:(batch_size, seq_max_len, head*v_size)   Second_attention_mask的作用在于：1）遮盖encoder序列的pad 2)遮盖decoder Q到每个词后面的词
        z = self.z_linear2(z) #z:(batch_size, seq_max_len, embedding_size)
        output2 = self.addnorm2(output1 + z) #output2:(batch_size, seq_max_len, embedding_size)

        # Feed Forward
        z = self.feedforward(output2) #z:(batch_size, seq_max_len, embedding_size)
        output = self.addnorm3(z + output2) #output:(batch_size, seq_max_len, embedding_size)
        return output
    
