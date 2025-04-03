# model.py
import torch
import torch.nn as nn
import math

# 设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, d_model=512, nhead=8, 
                 num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.trg_embedding = nn.Embedding(trg_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.fc_out = nn.Linear(d_model, trg_vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, trg):
        src_embed = self.src_embedding(src) * math.sqrt(self.d_model)
        trg_embed = self.trg_embedding(trg) * math.sqrt(self.d_model)
        src_embed = self.pos_encoder(src_embed)
        trg_embed = self.pos_encoder(trg_embed)
        src_embed = src_embed.permute(1, 0, 2)
        trg_embed = trg_embed.permute(1, 0, 2)
        src_padding_mask = (src == 0).to(device)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_embed.size(0)).to(device)
        output = self.transformer(
            src=src_embed,
            tgt=trg_embed,
            src_key_padding_mask=src_padding_mask,
            tgt_mask=trg_mask,
            memory_key_padding_mask=src_padding_mask
        )
        output = self.fc_out(output).permute(1, 0, 2)
        return output

    def encode(self, src):
        src_embed = self.src_embedding(src) * math.sqrt(self.d_model)
        src_embed = self.pos_encoder(src_embed)
        src_embed = src_embed.permute(1, 0, 2)
        memory = self.transformer.encoder(src_embed, src_key_padding_mask=(src == 0).to(device))
        return memory

    def decode(self, trg, memory):
        trg_embed = self.trg_embedding(trg) * math.sqrt(self.d_model)
        trg_embed = self.pos_encoder(trg_embed)
        trg_embed = trg_embed.permute(1, 0, 2)
        output = self.transformer.decoder(trg_embed, memory,
                                        tgt_mask=self.transformer.generate_square_subsequent_mask(trg.size(1)).to(device))
        output = self.fc_out(output).permute(1, 0, 2)
        return output