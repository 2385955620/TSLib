import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np


class Model(nn.Module):
    """
    iTransformer 修改版模型封装。
    参考论文: https://arxiv.org/abs/2310.06625

    该类根据不同任务（预测、插补、异常检测、分类）提供不同的前向计算分支。
    主要模块：
    - 输入嵌入（DataEmbedding_inverted）
    - 编码器（多层 Transformer EncoderLayer）
    - 根据任务选择的线性投影头（projection）或分类头
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        # 基本配置保存
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # ---- Embedding ----
        # 使用反归一化/嵌入层，将原始时间序列特征与时间标记信息一起编码为模型输入表示
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout
        )

        # ---- Encoder ----
        # 构建由多个 EncoderLayer 组成的 Encoder，每层包含自注意力和前馈网络
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model,
                        configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # ---- Projection / Head ----
        # 根据不同任务设置不同形状的线性投影层：
        # - 预测任务输出维度为 pred_len（在后面会调整维度顺序）
        # - 插补/异常检测任务输出为 seq_len（重建输入序列长度）
        # - 分类任务将编码表示展平并投影到类别数
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        if self.task_name == 'classification':
            # 分类任务使用非线性（gelu）+ dropout，再将所有通道拼接后线性映射到类别数
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        预测任务（long/short term forecast）的前向计算。

        参数:
        - x_enc: 原始输入序列，形状 [B, L, C]
        - x_mark_enc: 与编码器输入对应的时间特征（如时间戳编码）

        处理步骤：
        1. 对输入按时间步做均值/标准差归一化（参考 Non-stationary Transformer 的做法）
        2. 通过嵌入层 + 编码器提取表示
        3. 线性投影并调整维度得到预测结果
        4. 使用保存的均值与标准差对结果做反归一化
        """
        # ---- Normalization（归一化） ----
        # 记录均值用于反归一化；detach 防止梯度流回均值计算
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        # 计算标准差并避免除零（加上 1e-5）
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape

        # ---- Embedding + Encoder ----
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        print("encoder shape:",enc_out.shape)

        # 投影并调整为 [B, pred_len, N]；注意 projection 输出的第二维是 pred_len
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]

        print("MLP shape:",dec_out.shape)

        # ---- De-Normalization（反归一化） ----
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        """
        插补任务的前向计算（对缺失值进行重建）。

        过程与 forecast 类似，但输出长度为原始序列长度 seq_len（或 L）。
        """
        # ---- Normalization（归一化） ----
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, L, N = x_enc.shape

        # ---- Embedding + Encoder ----
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # 投影并调整为 [B, L, N]
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]

        # ---- De-Normalization（反归一化） ----
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        """
        异常检测任务的前向计算。

        仅使用编码器对输入序列进行重构，然后根据重构误差判断异常。
        """
        # ---- Normalization（归一化） ----
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, L, N = x_enc.shape

        # ---- Embedding + Encoder ----
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # 投影并反归一化得到重构序列
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        """
        分类任务的前向计算。

        将编码器输出经过非线性 + dropout，再展平后线性映射到类别概率（或 logits）。
        """
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # 非线性与 dropout
        output = self.act(enc_out)  # Transformer 的输出本身不包含激活函数
        output = self.dropout(output)
        # 展平为 (batch_size, enc_in * d_model)
        output = output.reshape(output.shape[0], -1)
        # 最终线性投影到类别数
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        根据任务选择对应的前向分支并返回结果。

        返回值形状说明：
        - 预测任务: [B, L, D]
        - 插补/异常检测: [B, L, D]
        - 分类: [B, N]
        """
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            # 返回最后 pred_len 步的预测，形状 [B, L, D]
            return dec_out[:, -self.pred_len:, :]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out
        return None
