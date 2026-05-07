import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math


# ==================== 1. 模型定义 ====================

class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class DecoderOnlyLayer(nn.Module):
    """仅解码器层（带因果掩码）"""

    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True, dropout=0.1)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_ff, d_model),
            nn.Dropout(0.1)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        attn_output, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_output)
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        return x


class DecoderOnlyModel(nn.Module):
    """仅解码器模型（GPT风格）"""

    def __init__(self, vocab_size, d_model=64, num_layers=3, num_heads=4, d_ff=256, max_len=100):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_encoding = PositionalEncoding(d_model, max_len)

        self.layers = nn.ModuleList([
            DecoderOnlyLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.output_layer = nn.Linear(d_model, vocab_size)

        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        batch_size, seq_len = x.shape
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
        mask = mask.to(x.device)

        x = self.token_embedding(x)
        x = self.position_encoding(x)

        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)
        logits = self.output_layer(x)
        return logits

    def generate(self, input_ids, max_new_tokens=5, temperature=0.8):
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self(input_ids)
                next_token_logits = logits[:, -1, :] / temperature
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids


# ==================== 2. 数据准备（修复词表缺失问题）====================

def create_vocabulary():
    """创建完整的词表"""
    tokens = [
        '<pad>', '<unk>',
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        '+', '-', '=', '?',
        '如果', '那么', '所以', '因为', '因此',
        '苹果', '香蕉', '水果', '是',
        '猫', '狗', '动物',
        '下雨', '地湿', '现在',  # 添加缺失的词汇
        '，', '。'
    ]

    word2idx = {word: idx for idx, word in enumerate(tokens)}
    idx2word = {idx: word for word, idx in word2idx.items()}

    return word2idx, idx2word, len(tokens)


def create_training_data(word2idx):
    """创建训练数据"""
    training_data = []

    # 加法训练
    additions = [
        (["1", "+", "1", "="], ["1", "+", "1", "=", "2"]),
        (["1", "+", "2", "="], ["1", "+", "2", "=", "3"]),
        (["2", "+", "2", "="], ["2", "+", "2", "=", "4"]),
        (["2", "+", "3", "="], ["2", "+", "3", "=", "5"]),
        (["3", "+", "3", "="], ["3", "+", "3", "=", "6"]),
        (["1", "+", "3", "="], ["1", "+", "3", "=", "4"]),
        (["2", "+", "4", "="], ["2", "+", "4", "=", "6"]),
    ]

    # 逻辑推理训练
    logic = [
        (["苹果", "是", "水果", "，", "香蕉", "是", "水果", "，", "所以"],
         ["苹果", "是", "水果", "，", "香蕉", "是", "水果", "，", "所以", "水果"]),
        (["猫", "是", "动物", "，", "狗", "是", "动物", "，", "所以"],
         ["猫", "是", "动物", "，", "狗", "是", "动物", "，", "所以", "动物"]),
        (["如果", "下雨", "，", "那么", "地湿", "。", "现在", "下雨", "，", "所以"],
         ["如果", "下雨", "，", "那么", "地湿", "。", "现在", "下雨", "，", "所以", "地湿"]),
    ]

    # 模式识别训练
    patterns = [
        (["1", "，", "2", "，", "3", "，", "4", "，"],
         ["1", "，", "2", "，", "3", "，", "4", "，", "5"]),
        (["2", "，", "4", "，", "6", "，", "8", "，"],
         ["2", "，", "4", "，", "6", "，", "8", "，", "10"]),
    ]

    all_examples = additions + logic + patterns

    for input_tokens, target_tokens in all_examples:
        try:
            input_ids = [word2idx[t] for t in input_tokens]
            target_ids = [word2idx[t] for t in target_tokens]
            training_data.append((input_ids, target_ids))
        except KeyError as e:
            print(f"警告：跳过包含未知词的样本: {e}")
            continue

    return training_data


def create_test_data(word2idx):
    """创建测试数据"""
    test_cases = [
        (["1", "+", "1", "="], "2", "加法：1+1=2"),
        (["2", "+", "3", "="], "5", "加法：2+3=5"),
        (["1", "+", "2", "="], "3", "加法：1+2=3"),
        (["苹果", "是", "水果", "，", "香蕉", "是", "水果", "，", "所以"], "水果", "逻辑推理：苹果和香蕉都是水果"),
        (["猫", "是", "动物", "，", "狗", "是", "动物", "，", "所以"], "动物", "逻辑推理：猫和狗都是动物"),
        (["如果", "下雨", "，", "那么", "地湿", "。", "现在", "下雨", "，", "所以"], "地湿", "条件推理：下雨则地湿"),
        (["1", "，", "2", "，", "3", "，", "4", "，"], "5", "数列续写：1,2,3,4..."),
    ]

    test_data = []
    for tokens, expected, desc in test_cases:
        try:
            input_ids = [word2idx[t] for t in tokens]
            expected_id = word2idx[expected]
            test_data.append((input_ids, expected_id, desc))
        except KeyError as e:
            print(f"警告：测试样本包含未知词: {e}")
            continue

    return test_data


# ==================== 3. 训练 ====================

def run_training(model, training_data, epochs=150):
    """训练模型"""
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    print("\n" + "=" * 80)
    print("开始训练")
    print("=" * 80)

    losses = []
    for epoch in range(epochs):
        total_loss = 0
        for input_ids, target_ids in training_data:
            input_tensor = torch.tensor([input_ids])
            target_tensor = torch.tensor([target_ids])

            optimizer.zero_grad()
            output = model(input_tensor)

            # 只计算最后一个位置的损失
            output_last = output[:, -1, :]
            target_last = target_tensor[:, -1]
            loss = criterion(output_last, target_last)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(training_data)
        losses.append(avg_loss)

        if (epoch + 1) % 30 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

    print("训练完成！")
    return losses


# ==================== 4. 测试 ====================

def run_evaluation(model, test_data, idx2word):
    """评估模型"""
    model.eval()

    print("\n" + "=" * 80)
    print("测试结果")
    print("=" * 80)

    correct = 0
    with torch.no_grad():
        for i, (input_ids, expected_id, desc) in enumerate(test_data, 1):
            input_tensor = torch.tensor([input_ids])
            output = model(input_tensor)
            next_token_id = torch.argmax(output[:, -1, :], dim=-1).item()

            input_text = ' '.join([idx2word.get(idx, '?') for idx in input_ids])
            predicted = idx2word.get(next_token_id, '?')
            expected = idx2word.get(expected_id, '?')

            print(f"\n测试{i}: {desc}")
            print(f"  输入: {input_text}")
            print(f"  预测: {predicted}")
            print(f"  期望: {expected}")

            if next_token_id == expected_id:
                print("  ✅ 正确")
                correct += 1
            else:
                print("  ❌ 错误")

    print(f"\n准确率: {correct}/{len(test_data)} ({correct / len(test_data) * 100:.1f}%)")
    return correct / len(test_data)


# ==================== 5. 生成 ====================

def run_generation(model, prompts, idx2word, word2idx):
    """文本生成"""
    model.eval()

    print("\n" + "=" * 80)
    print("文本生成")
    print("=" * 80)

    for prompt in prompts:
        print(f"\n输入: {' '.join(prompt)}")

        # 检查所有词是否在词表中
        valid_prompt = []
        for t in prompt:
            if t in word2idx:
                valid_prompt.append(t)
            else:
                print(f"  警告: '{t}' 不在词表中，跳过")
                valid_prompt.append('<unk>')

        input_ids = [word2idx.get(t, word2idx['<unk>']) for t in valid_prompt]
        input_tensor = torch.tensor([input_ids])

        with torch.no_grad():
            generated = input_ids.copy()
            for step in range(5):
                output = model(torch.tensor([generated]))
                next_id = torch.argmax(output[:, -1, :], dim=-1).item()
                next_word = idx2word.get(next_id, '?')
                generated.append(next_id)
                print(f"  生成{step + 1}: {next_word}")
                if next_word in ['。', '<end>']:
                    break

        full_text = ' '.join([idx2word.get(idx, '?') for idx in generated])
        print(f"完整: {full_text}")


# ==================== 6. 主程序 ====================

def main():
    print("=" * 80)
    print("仅解码器模型 - 训练推理演示")
    print("=" * 80)

    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 创建词表
    word2idx, idx2word, vocab_size = create_vocabulary()
    print(f"\n词表大小: {vocab_size}")
    print(f"词表示例: {list(word2idx.keys())[:20]}...")

    # 创建数据
    training_data = create_training_data(word2idx)
    test_data = create_test_data(word2idx)

    if not training_data:
        print("错误：没有有效的训练数据！")
        return

    print(f"\n训练样本数: {len(training_data)}")
    print(f"测试样本数: {len(test_data)}")

    # 创建模型
    model = DecoderOnlyModel(
        vocab_size=vocab_size,
        d_model=64,
        num_layers=3,
        num_heads=4,
        d_ff=256
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    # 训练
    print("\n开始训练...")
    losses = run_training(model, training_data, epochs=150)

    # 测试
    if test_data:
        accuracy = run_evaluation(model, test_data, idx2word)
    else:
        print("没有测试数据")

    # 生成
    prompts = [
        ["1", "+", "1", "="],
        ["2", "+", "2", "="],
        ["苹果", "是", "水果", "，", "香蕉", "是", "水果", "，", "所以"],
        ["如果", "下雨", "，", "那么", "地湿", "。", "现在", "下雨", "，", "所以"],
        ["1", "，", "2", "，", "3", "，", "4", "，"],
    ]
    run_generation(model, prompts, idx2word, word2idx)

    # 绘制损失曲线（可选）
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.grid(True)
        plt.savefig('training_loss.png')
        print("\n损失曲线已保存为 training_loss.png")
    except ImportError:
        pass
    except Exception as e:
        print(f"\n绘图失败: {e}")

    print("\n" + "=" * 80)
    print("完成！")
    print("=" * 80)


# 直接运行
if __name__ == "__main__":
    main()