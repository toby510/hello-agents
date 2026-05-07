import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import copy


# ==================== 模型定义 ====================

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制模块
    """

    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 定义 Q, K, V 和输出的线性变换层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.debug = False

    def set_debug(self, debug):
        self.debug = debug
    """
    缩放点积注意力：
    1-缩放：Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)，结果除以d_k，避免数据进入softmax饱和
    2-点积：对应 Q・Kᵀ，计算相似度
    3-注意力 —— 对应 Softmax
    """
    def scaled_dot_product_attention(self, Q, K, V, mask=None, layer_name=""):
        # 1. 计算注意力得分 (QK^T)，得到的是每个单词之间的相似度，即注意力分数，形状为n*n,n为单词数量。得到的是细节如下：
        # 1)K.transpose(-2, -1)=K最后两维转置，为了矩阵乘法；
        # 2)math.sqrt(self.d_k)=缩放，防止数值爆炸
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 2. 应用掩码 (如果提供)。masked_fill 把所有不可见位置的分数改成 -1e9，这样掩码的地方原始attn_scores会被更改为-1e9,最终执行softmax的时候概率为0，这样模型就“看不见”
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # 3. 计算注意力权重 (Softmax)
        attn_probs = torch.softmax(attn_scores, dim=-1)

        if self.debug:
            print(f"{layer_name} 注意力权重示例（第1个头，前3个位置）:")
            print(attn_probs[0, 0, :3, :3])

        # 4. 加权求和 (权重 * V)，计算之前要先把注意力转化为概率，否则没有意义，因为attn_scores是无界裸分：可正、可负、差距极大、总和乱飘，算概率就是要给一个有界值，总和=1
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        # 将输入 x 的形状从 (batch_size, seq_length, d_model)
        # 变换为 (batch_size, num_heads, seq_length, d_k)
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        # 将输入 x 的形状从 (batch_size, num_heads, seq_length, d_k)
        # 变回 (batch_size, seq_length, d_model)
        batch_size, num_heads, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None, layer_name=""):
        # 1. 对 Q, K, V 进行线性变换，多头思想是：多次并行执行注意力机制，具体如下
        # 拆分多头后，Q、K、V的形状发了改变，多了heads维度，每个Q/K/V都有heads组子集，每个子集的维度=总维度/heads
        # 比如原来是；(1,9,64),改成4个头后变成：(1,4,9,16),这里的4表示9个单词，每个单词都有4个头，可以理解有4个Q、K、V，分别为[0][0],[0][1],[0][2],[0][3]
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        # 2. 计算缩放点积注意力
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask, layer_name)

        # 3. 把多头分开算的结果，重新拼回一个完整向量===>然后进行最终的线性变换
        #线性变换原因：W_o融合多头信息（融合 = 提炼）
        output = self.W_o(self.combine_heads(attn_output))
        return output


class PositionWiseFeedForward(nn.Module):
    """
    位置前馈网络模块
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class PositionalEncoding(nn.Module):
    """
    为输入序列的词嵌入向量添加位置编码。
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建一个足够长的位置编码矩阵
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class EncoderLayer(nn.Module):
    """
    编码器核心层
    """

    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask, layer_name=""):
        # 1. 多头自注意力,x=embedding+pos后的向量，self_attn自我注意力，自身上下文的理解
        attn_output = self.self_attn(x, x, x, mask, f"{layer_name}_self_attn")
        # 2. dropout->参差处理->再归一化
        #1）Dropout(0.1) 意思：随机扔掉 10% 的神经元随机选一个位置，直接变成 0，其他不变，比如原来attn_output = [0.5, 1.2, -0.3, 2.1, -1.5, 0.8]
             # dropout后 = [0.5, 0.0, -0.3, 2.1, -1.5, 0.8]，作用：随机破坏一部分特征，强迫模型学得更健壮、更通用！避免死记硬背
        #2)参差处理:保持原来输入x的特征
        #3)归一化：保证数据在一定的范围，不要偏离太大，锁死在稳定分布，让数据在【均值=0，方差=1的范围内】
        x = self.norm1(x + self.dropout(attn_output))

        # 2. 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class DecoderLayer(nn.Module):
    """
    解码器核心层
    """

    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask, tgt_mask, layer_name=""):
        # 1. 掩码多头自注意力 (对自己)，x为target，encoder_output为编码器输出
        attn_output = self.self_attn(x, x, x, tgt_mask, f"{layer_name}_self_attn")
        # 2. x + self.dropout(attn_output)=残差连接,把原来的信息x保留下来，加上注意力学到的新信息.模型堆叠很多层（6 层、12 层、24 层）后，梯度会消失 → 前面的层完全学不到东西！有了残差：新x = 原来的x + 新学到的特征,梯度可以直接沿着 x 这条捷径传回前面，不会消失！
        # 3. self.norm1=层归一化。需要的原因：注意力输出的值 可大可小、不稳定；堆叠多层后，数值会 越来越大 / 越来越小 → 训练崩溃，LayerNorm 做的事：把每个词向量的数值强行规范成：均值 0，方差 1
        x = self.norm1(x + self.dropout(attn_output))

        # 4. 交叉注意力 (对编码器输出)：Q=target，K=encoder_output,V=encoder_output，表示从编码器中找出跟target对应的src
        cross_attn_output = self.cross_attn(x, encoder_output, encoder_output, src_mask, f"{layer_name}_cross_attn")
        # 5. 残差+归一化
        x = self.norm2(x + self.dropout(cross_attn_output))

        # 6. 前馈网络
        ff_output = self.feed_forward(x)

        # 7.残差+归一化
        x = self.norm3(x + self.dropout(ff_output))

        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_len):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask, debug=True):
        # 设置调试模式
        for layer in self.layers:
            layer.self_attn.set_debug(debug)

        if debug:
            print("\n" + "=" * 60)
            print("编码器处理开始")
            print("=" * 60)
            print(f"\n1. 词嵌入后形状: {x.shape}")

        x = self.embedding(x)
        x = self.pos_encoder(x)

        if debug:
            print(f"2. 位置编码后形状: {x.shape}")

        for i, layer in enumerate(self.layers):
            if debug:
                print(f"\n--- 编码器第 {i + 1} 层 ---")
            x = layer(x, mask, f"encoder_layer_{i + 1}")
            if debug and i == 0:
                print(f"   第{i + 1}层输出示例（前2个token的前5维）:")
                print(f"   {x[0, :2, :5]}")

        x = self.norm(x)
        if debug:
            print(f"\n3. 编码器最终输出形状: {x.shape}")
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_len):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output, src_mask, tgt_mask, debug=False):
        # 设置调试模式
        for layer in self.layers:
            layer.self_attn.set_debug(debug)
            layer.cross_attn.set_debug(debug)

        if debug:
            print("\n" + "=" * 60)
            print("解码器处理开始")
            print("=" * 60)
            print(f"\n1. 词嵌入后形状: {x.shape}")

        x = self.embedding(x)
        x = self.pos_encoder(x)

        if debug:
            print(f"2. 位置编码后形状: {x.shape}")

        for i, layer in enumerate(self.layers):
            if debug:
                print(f"\n--- 解码器第 {i + 1} 层 ---")
            x = layer(x, encoder_output, src_mask, tgt_mask, f"decoder_layer_{i + 1}")
            if debug and i == 0:
                print(f"   第{i + 1}层输出示例（前2个token的前5维）:")
                print(f"   {x[0, :2, :5]}")

        x = self.norm(x)
        if debug:
            print(f"\n3. 解码器最终输出形状: {x.shape}")
        return x


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_len=5000):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_len)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_len)
        self.final_linear = nn.Linear(d_model, tgt_vocab_size)

    def generate_mask(self, src, tgt):
        # src_mask: (batch_size, 1, 1, src_len)
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)

        # tgt_mask: (batch_size, 1, tgt_len, tgt_len)
        tgt_pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        tgt_len = tgt.size(1)
        tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=src.device)).bool()
        tgt_mask = tgt_pad_mask & tgt_sub_mask

        return src_mask, tgt_mask

    def forward(self, src, tgt, debug=True):
        src_mask, tgt_mask = self.generate_mask(src, tgt)

        encoder_output = self.encoder(src, src_mask, debug)
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask, debug)

        output = self.final_linear(decoder_output)
        return output


# ==================== 训练和测试代码 ====================

def create_vocab():
    """创建词表"""
    src_vocab = {
        0: "<pad>",
        1: "The", 2: "cat", 3: "sat", 4: "on", 5: "the", 6: "mat",
        7: "A", 8: "dog", 9: "ran", 10: "under", 11: "car"
    }

    tgt_vocab = {
        0: "<pad>",
        1: "<start>",
        2: "<end>",
        3: "熊", 4: "坐", 5: "在", 6: "垫子", 7: "上",
        8: "一只", 9: "狗", 10: "跑", 11: "下面", 12: "车"
    }

    # 反向映射
    src_word_to_idx = {v: k for k, v in src_vocab.items()}
    tgt_word_to_idx = {v: k for k, v in tgt_vocab.items()}

    return src_vocab, tgt_vocab, src_word_to_idx, tgt_word_to_idx


def pad_sequence(seq, max_len, pad_token=0):
    """对序列进行padding"""
    if len(seq) >= max_len:
        return seq[:max_len]
    return seq + [pad_token] * (max_len - len(seq))


def create_training_data(src_word_to_idx, tgt_word_to_idx):
    """创建训练数据"""
    training_data = []

    # 训练样本1: The cat sat on the mat → 猫 坐 在 垫子 上
    src1 = [src_word_to_idx[w] for w in ["The", "cat", "sat", "on", "the", "mat"]]
    tgt1 = [tgt_word_to_idx["<start>"],
            tgt_word_to_idx["熊"],
            tgt_word_to_idx["坐"],
            tgt_word_to_idx["在"],
            tgt_word_to_idx["垫子"],
            tgt_word_to_idx["上"],
            tgt_word_to_idx["<end>"]]
    training_data.append((src1, tgt1))

    # 训练样本2: A dog ran under car → 一只 狗 跑 下面 车
    src2 = [src_word_to_idx[w] for w in ["A", "dog", "ran", "under", "car"]]
    tgt2 = [tgt_word_to_idx["<start>"],
            tgt_word_to_idx["一只"],
            tgt_word_to_idx["狗"],
            tgt_word_to_idx["跑"],
            tgt_word_to_idx["下面"],
            tgt_word_to_idx["车"],
            tgt_word_to_idx["<end>"]]
    training_data.append((src2, tgt2))

    return training_data


def train_transformer(model, training_data, epochs=200, verbose=True):
    """训练Transformer模型"""
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略padding

    if verbose:
        print("\n" + "=" * 80)
        print("开始训练 Transformer")
        print("=" * 80)

    losses = []
    for epoch in range(epochs):
        total_loss = 0

        for src_seq, tgt_seq in training_data:
            # 准备数据（padding到相同长度）
            max_len = max(len(src_seq), len(tgt_seq)) + 2
            src = torch.tensor([pad_sequence(src_seq, max_len)])
            tgt = torch.tensor([pad_sequence(tgt_seq, max_len)])

            # 前向传播
            optimizer.zero_grad()
            output = model(src, tgt, debug=False)

            # 计算损失（忽略padding和<start>）
            output = output[:, :-1, :].reshape(-1, output.size(-1))
            target = tgt[:, 1:].reshape(-1)
            loss = criterion(output, target)

            # 反向传播
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(training_data)
        losses.append(avg_loss)

        # 每10个epoch打印一次损失
        if verbose and (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

    if verbose:
        print("\n训练完成！")

    return losses


def translate(model, src_sentence, src_word_to_idx, tgt_vocab, max_len=10):
    """使用训练好的模型进行翻译"""
    model.eval()

    with torch.no_grad():
        # 将源句子转换为索引
        src_indices = []
        words = src_sentence.split()
        for word in words:
            if word in src_word_to_idx:
                src_indices.append(src_word_to_idx[word])
            else:
                print(f"警告：词 '{word}' 不在词表中")
                return None

        # 不足max_len时，做填充0处理，即Padding
        src = torch.tensor([pad_sequence(src_indices, max_len)])

        # 对src进行掩码
        # 1）编码时不让模型把注意力分给 <pad>，掩码最终效果：[ [ [ [True, True, True, True, True, True, False, False, False] ] ] ]，False的在计算self_attn时会被掩码掉
        # 2）解码时不让中文词去关注英文的 <pad>，保证翻译只对齐真实有效单词，不匹配无效填充位
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        # 对src进行编码，二期是掩码编码，掩码对象是padding的部分
        encoder_output = model.encoder(src, src_mask, debug=False)

        print(f"\n编码器输出: {encoder_output}")
        print(f"\n源句子: {src_sentence}")
        print("翻译中...")

        # 逐步生成
        generated = []
        #torch.tensor([[1]])里的1是word的idx,即"<start>",给解码器喂第一个 “启动信号”：<start> 符号！ 告诉模型：请从这里开始生成中文翻译
        tgt_input = torch.tensor([[1]])  # <start>

        for step in range(max_len):
            # 生成掩码
            src_mask, tgt_mask = model.generate_mask(src, tgt_input)

            # 解码
            decoder_output = model.decoder(tgt_input, encoder_output, src_mask, tgt_mask, debug=False)

            #decoder_output代表每个位置的语义特征向量（64 维数字），所以需要转成词汇表，最终每一行指向一个词的概率
            logits = model.final_linear(decoder_output)

            #只取最后一个位置的分数！因为推理是逐字生成，我们只需要最后一个位置预测下一个词
            next_token_logits = logits[:, -1, :]
            #把分数转成概率
            next_token_probs = torch.softmax(next_token_logits, dim=-1)
            #取概率最大值对应的索引
            next_token_id = torch.argmax(next_token_probs, dim=-1).item()

            # 获取预测的词
            next_word = tgt_vocab.get(next_token_id, f"<unk>")
            # 推理到的词汇的概率
            confidence = next_token_probs[0, next_token_id].item()

            # 如果是结束标记，停止
            if next_word == "<end>" or step == max_len - 1:
                break

            generated.append(next_word)

            # 更新输入
            new_token = torch.tensor([[next_token_id]])
            # 拼接已推理的词作为新的输入，来预测下一个词
            tgt_input = torch.cat([tgt_input, new_token], dim=1)

        translation = ' '.join(generated)
        print(f"翻译结果: {translation}")
        print(f"最后一步置信度: {confidence:.3f}")

        return translation


def visualize_attention_demo(model, src_sentence, src_word_to_idx, tgt_vocab):
    """演示注意力机制（简化版）"""
    model.eval()

    with torch.no_grad():
        # 准备源序列
        src_indices = [src_word_to_idx[w] for w in src_sentence.split()]
        src = torch.tensor([pad_sequence(src_indices, max_len=10)])

        # 编码
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        encoder_output = model.encoder(src, src_mask, debug=False)

        print("\n" + "=" * 80)
        print("注意力机制演示")
        print("=" * 80)
        print(f"\n源句子: {src_sentence}")
        print("\n注意：由于模型是在小数据集上训练的，")
        print("注意力权重可能不够理想。")
        print("充分训练后，你会看到每个目标词聚焦到对应的源词。")


# ==================== 主程序 ====================

if __name__ == "__main__":
    # 设置参数
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    d_model = 64  # 词嵌入维度
    num_layers = 3  # 编码器/解码器层数
    num_heads = 4  # 注意力头数
    d_ff = 256  # 前馈网络维度
    dropout = 0.1
    max_len = 20

    # 创建词表
    src_vocab, tgt_vocab, src_word_to_idx, tgt_word_to_idx = create_vocab()

    print("=" * 80)
    print("Transformer 机器翻译模型")
    print("=" * 80)
    print(f"模型配置:")
    print(f"  - 词嵌入维度: {d_model}")
    print(f"  - 编码器层数: {num_layers}")
    print(f"  - 解码器层数: {num_layers}")
    print(f"  - 注意力头数: {num_heads}")

    # 创建模型（先创建模型）
    model = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_layers,
                        num_heads, d_ff, dropout, max_len)

    # 计算并打印参数量（在创建模型之后）
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - 总参数量: {total_params:,}")
    print(f"  - 可训练参数量: {trainable_params:,}")

    # 创建训练数据
    training_data = create_training_data(src_word_to_idx, tgt_word_to_idx)
    print(f"\n训练样本数: {len(training_data)}")
    print(f"\n训练样本明细: {training_data}")

    # 训练模型
    print("\n开始训练...")
    losses = train_transformer(model, training_data, epochs=200, verbose=True)

    # 测试翻译
    print("\n" + "🎯" * 40)
    print("测试翻译")
    print("🎯" * 40)

    # 测试训练过的句子
    translate(model, "The cat sat on the mat", src_word_to_idx, tgt_vocab)
    translate(model, "A dog ran under car", src_word_to_idx, tgt_vocab)

    # 演示注意力机制
    visualize_attention_demo(model, "The cat sat on the mat", src_word_to_idx, tgt_vocab)

    print("\n" + "=" * 80)
    print("完成！")
    print("=" * 80)