import torch
import torch.nn as nn
from embedding_posittional_encoding import Token_and_Position_embedding


class Transformer_Encoder_Block(nn.Module):
    def __init__(self, embedding_dim, num_heads, ff_dim, 
                 dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embedding_dim, 
                                          num_heads,
                                          batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim, bias=True),
            nn.ReLU(),
            nn.Linear(ff_dim, embedding_dim, bias=True)
        )
        self.layernorm = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.dropout = nn.Dropout( dropout )

    def forward(self, q, k, v):
        attn_output, _ = self.attn(q, k, v)
        attn_output = self.dropout( attn_output )

        out_1 = self.layernorm( q + attn_output )
        ffn_ouput = self.ffn( out_1 )
        ffn_ouput = self.dropout( ffn_ouput )

        out_2 = self.layernorm( out_1 + ffn_ouput )
        return out_2
    
class Transformer_Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_length, 
                 num_layers, num_heads, ff_dim, 
                 dropout=0.1, device="cpu" ):
        super().__init__()
        self.embedding = Token_and_Position_embedding(vocab_size= vocab_size,
                                                      embedding_dim= embedding_dim,
                                                      max_length= max_length,
                                                      device= device)
        
        self.layers = nn.ModuleList(
            [Transformer_Encoder_Block(embedding_dim= embedding_dim,
                                       num_heads= num_heads,
                                       ff_dim= ff_dim,
                                       dropout= dropout) for i in range(num_layers)]
        )
    
    def forward(self, x):
        output = self.embedding( x )
        for layer in self.layers:
            output = layer( output, output, output )
        
        return output