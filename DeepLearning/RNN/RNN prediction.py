import torch
from torch import nn


# ==========================================
# A. 必须重新定义一遍模型结构 (或者从你的 train.py import)
# ==========================================
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(input_size=embed_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, state):
        X = self.embedding(x)
        Y, state = self.rnn(X, state)
        output = self.linear(Y.reshape(-1, Y.shape[-1]))
        return output, state

    def begin_state(self, batch_size, device):
        return torch.zeros((self.num_layers, batch_size, self.hidden_size), device=device)


# 预测函数也需要
def predict(prefix, num_preds, model, vocab_idx, idx_vocab, device):
    state = model.begin_state(batch_size=1, device=device)
    outputs = [vocab_idx[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape(1, 1)
    for y in prefix[1:]:
        _, state = model(get_input(), state)
        outputs.append(vocab_idx[y])
    for _ in range(num_preds):
        y, state = model(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([idx_vocab[i] for i in outputs])


# ==========================================
# B. 加载过程
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
path = "rnn_time_machine.pth"

if __name__ == '__main__':
    print("正在加载模型...")
    # 1. 读取文件
    checkpoint = torch.load(path, map_location=device)

    # 2. 恢复词表 (这是最关键的一步！)
    vocab_data = checkpoint['vocab']
    token_to_idx = vocab_data['token_to_idx']
    idx_to_token = vocab_data['idx_to_token']

    # 3. 恢复超参数
    params = checkpoint['hyper_params']

    # 4. 实例化模型 (使用保存的超参数，而不是硬编码)
    loaded_model = RNNModel(
        vocab_size=params['vocab_size'],
        embed_size=params['embed_size'],
        hidden_size=params['hidden_size'],
        num_layers=params['num_layers']
    ).to(device)

    # 5. 加载权重
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model.eval()  # 切换到预测模式 (关闭Dropout等)

    print("模型加载成功！")

    # ==========================================
    # C. 开始预测
    # ==========================================
    # 咱们来试试让它写一段话
    start_text = "time traveller"
    generated_text = predict(start_text, 100, loaded_model, token_to_idx, idx_to_token, device)

    print("-" * 30)
    print(f"输入: {start_text}")
    print(f"续写: {generated_text}")
    print("-" * 30)