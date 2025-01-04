import torch
import torch.nn as nn

class Token_and_Position_embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, 
                 max_length, device="cpu"):
        super().__init__()
        self.device = device
        self.text_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embedding = nn.Embedding(max_length, embedding_dim)
    
    def forward(self, x):
        N, seq_len = x.size()
        positons = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)
        output_1 = self.text_embedding( x )
        output_2 = self.pos_embedding( positons )
        return output_1 + output_2