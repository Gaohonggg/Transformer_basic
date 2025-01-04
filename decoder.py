import torch
import torch.nn as nn
from embedding_posittional_encoding import Token_and_Position_embedding


class Transformer_Decoder_Block(nn.Module):
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

    def forward(self, x, enc_output, src_mark, tgt_mask):
        attn_output, _ = self.attn(x, x, x, 
                                   attn_mask= tgt_mask)
        attn_output = self.dropout( attn_output )
        
        output_1 = self.layernorm( x + attn_output )
        attn_output, _ = self.attn(output_1, enc_output, enc_output,
                                   attn_mask= src_mark)
        attn_output = self.dropout( attn_output )

        output_2 = self.layernorm( output_1 + attn_output )
        ffn_output = self.ffn( output_2 )
        ffn_output = self.dropout( ffn_output )

        output_3 = self.layernorm( output_2 + ffn_output )

        return output_3
    
class Transformer_Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_length, 
                 num_layers, num_heads, ff_dim, 
                 dropout=0.1, device="cpu" ):
        super().__init__()
        self.embedding = Token_and_Position_embedding(vocab_size= vocab_size,
                                                      embedding_dim= embedding_dim,
                                                      max_length= max_length,
                                                      device= device)
        
        self.layers = nn.ModuleList(
            [Transformer_Decoder_Block(embedding_dim= embedding_dim,
                                       num_heads= num_heads,
                                       ff_dim= ff_dim,
                                       dropout= dropout) for i in range(num_layers)]
        )
    
    def forward(self, x, enc_output, src_mark, tgt_mask):
        output = self.embedding( x )
        for layer in self.layers:
            output = layer( output, enc_output, src_mark, tgt_mask)
        
        return output
