# translate.py
import torch
import re
from torchtext.data.utils import get_tokenizer
from model import TransformerModel

# 设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载词汇表
en_vocab = torch.load('en_vocab.pth')
zh_vocab = torch.load('zh_vocab.pth')

# 预处理函数
def preprocess_sentence(sentence, lang_type):
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    if lang_type == 'en':
        sentence = sentence.lower()
    return sentence.strip()

# 分词器
en_tokenizer = get_tokenizer('basic_english')
zh_tokenizer = lambda x: list(x.strip())

# 加载模型
model = TransformerModel(
    src_vocab_size=len(en_vocab),
    trg_vocab_size=len(zh_vocab)
).to(device)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

def translate(sentence):
    # 预处理
    processed_sent = preprocess_sentence(sentence, 'en')
    
    # 转换为模型输入
    en_tokens = ['<bos>'] + en_tokenizer(processed_sent) + ['<eos>']
    en_indices = [en_vocab[token] for token in en_tokens]
    src_tensor = torch.LongTensor(en_indices).unsqueeze(0).to(device)
    
    # 编码解码
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
    
    # 转换为中文
    translated_tokens = [zh_vocab.get_itos()[idx] for idx in trg_indices]
    return ''.join(translated_tokens[1:-1])  # 中文不需要空格分隔

if __name__ == "__main__":
    while True:
        text = input("请输入要翻译的英文句子（输入q退出）: ")
        if text.lower() == 'q':
            break
        print("翻译结果:", translate(text))
        print("-" * 50)