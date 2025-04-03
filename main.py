# 基于 Transformer 的中英文翻译模型（优化版）

# 导入必要的库
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math
import re

# 禁用 torchtext 的弃用警告
import torchtext
torchtext.disable_torchtext_deprecation_warning()

# 检查 GPU 并设置随机种子
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)

# ================== 数据预处理优化 ==================
def preprocess_sentence(sentence, lang_type):
    """数据清洗：统一小写、去除特殊字符、增加空格"""
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    if lang_type == 'en':
        sentence = sentence.lower()
    return sentence.strip()

# 读取并预处理数据
def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                en = preprocess_sentence(parts[0], 'en')
                zh = preprocess_sentence(parts[1], 'zh')
                data.append([en, zh])
    return pd.DataFrame(data, columns=['English', 'Chinese'])

# ================== 词汇表构建 ==================
# 定义分词器
en_tokenizer = get_tokenizer('basic_english')
zh_tokenizer = lambda x: list(x.strip())  # 中文按字符分词

# 构建词汇表
def yield_tokens(data_iter, lang_idx):
    for en, zh in data_iter:
        if lang_idx == 0:
            yield en_tokenizer(en)
        else:
            yield zh_tokenizer(zh)

# 加载数据
df = load_data('data/cmn.txt')

# 构建词汇表
SRC_VOCAB_SIZE = 10000
TRG_VOCAB_SIZE = 5000

en_vocab = build_vocab_from_iterator(
    yield_tokens(df.values, 0),
    max_tokens=SRC_VOCAB_SIZE,
    specials=['<pad>', '<unk>', '<bos>', '<eos>']
)
en_vocab.set_default_index(en_vocab['<unk>'])

zh_vocab = build_vocab_from_iterator(
    yield_tokens(df.values, 1),
    max_tokens=TRG_VOCAB_SIZE,
    specials=['<pad>', '<unk>', '<bos>', '<eos>']
)
zh_vocab.set_default_index(zh_vocab['<unk>'])

# ================== 数据集类定义 ==================
class TranslationDataset(Dataset):
    def __init__(self, df, en_vocab, zh_vocab):
        self.df = df
        self.en_vocab = en_vocab
        self.zh_vocab = zh_vocab

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        en_sent = self.df.iloc[idx]['English']
        zh_sent = self.df.iloc[idx]['Chinese']

        # 英文处理
        en_tokens = ['<bos>'] + en_tokenizer(en_sent) + ['<eos>']
        en_indices = [self.en_vocab[token] for token in en_tokens]
        
        # 中文处理
        zh_tokens = ['<bos>'] + zh_tokenizer(zh_sent) + ['<eos>']
        zh_indices = [self.zh_vocab[token] for token in zh_tokens]
        
        return torch.tensor(en_indices, dtype=torch.long), torch.tensor(zh_indices, dtype=torch.long)

# ================== 数据加载器 ==================
def collate_fn(batch):
    src_batch, trg_batch = [], []
    for src, trg in batch:
        src_batch.append(src)
        trg_batch.append(trg)
    src_batch = pad_sequence(src_batch, padding_value=en_vocab['<pad>'], batch_first=True)
    trg_batch = pad_sequence(trg_batch, padding_value=zh_vocab['<pad>'], batch_first=True)
    return src_batch, trg_batch

# 创建数据集和数据加载器
dataset = TranslationDataset(df, en_vocab, zh_vocab)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# ================== 模型架构优化 ==================
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
        
        # 调整维度到(seq_len, batch, d_model)
        src_embed = src_embed.permute(1, 0, 2)
        trg_embed = trg_embed.permute(1, 0, 2)
        
        # 生成mask
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

# ================== 训练优化 ==================
def train_epoch(model, loader, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for src, trg in loader:
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        output = model(src, trg[:, :-1])
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)

def evaluate(model, loader, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for src, trg in loader:
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg[:, :-1])
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(loader)

# ================== 主程序 ==================
if __name__ == "__main__":
    # 初始化模型
    model = TransformerModel(
        src_vocab_size=len(en_vocab),
        trg_vocab_size=len(zh_vocab)
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss(ignore_index=zh_vocab['<pad>'])
    
    # 训练循环
    best_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    for epoch in range(20):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, clip=1.0)
        val_loss = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)
        
        print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f}')
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print("Early stopping!")
            break

    # 加载最佳模型
    model.load_state_dict(torch.load('best_model.pth'))

    # 在原训练代码最后（model.load_state_dict之后）添加：
    torch.save(en_vocab, 'en_vocab.pth')
    torch.save(zh_vocab, 'zh_vocab.pth')
    
    # 测试翻译
    test_sentence = "I try"
    en_tokens = ['<bos>'] + en_tokenizer(preprocess_sentence(test_sentence, 'en')) + ['<eos>']
    en_indices = [en_vocab[token] for token in en_tokens]
    src_tensor = torch.LongTensor(en_indices).unsqueeze(0).to(device)
    
    # 生成翻译
    model.eval()
    with torch.no_grad():
        memory = model.encode(src_tensor)
        trg_indices = [zh_vocab['<bos>']]
        for _ in range(50):
            trg_tensor = torch.LongTensor(trg_indices).unsqueeze(0).to(device)
            output = model.decode(trg_tensor, memory)
            next_token = output.argmax(2)[:, -1].item()
            trg_indices.append(next_token)
            if next_token == zh_vocab['<eos>']:
                break
        
    translated_tokens = [zh_vocab.get_itos()[idx] for idx in trg_indices]
    print(' '.join(translated_tokens[1:-1]))  # 去除<bos>和<eos>