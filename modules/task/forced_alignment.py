import math

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
import matplotlib.pyplot as plt

from ..jit.forced_alignment import decode_matrix, generate_matrix, generate_prior


class ForcedAlignmentTask(nn.Module):
    """
    ForcedAlignmentTask负责对给定正确的普通音素序列进行对齐，不负责音素序列有误/缺少音素序列/SP等特殊音素的情况
    """

    def __init__(
        self,
        upsample_rate,  # 上采样音频特征，时间分辨率更高
        audio_network: nn.Module,  # 让音频特征的维度与音素特征的维度一致
        num_phones,
        phone_embedding_dims,  # 音素嵌入维度
        phone_network: nn.Module,  # 音素特征网络
    ):
        super().__init__()

        self.upsample_rate = upsample_rate
        self.upsample_layer = nn.Upsample(scale_factor=upsample_rate)
        self.audio_network = audio_network

        self.phone_embedding = nn.Embedding(num_phones, phone_embedding_dims)
        self.phone_prior = nn.Embedding(num_phones, 1)
        self.phone_prior.weight.data.zero_()  # TODO: 使用统计出来的数据
        self.phone_network = phone_network

        # 用于熵不变性softmax
        self.log_base = 25  # TODO: 使用统计出来的log长度的平均再exp
        self.pseudo_label_threshold = -2  # 自适应阈值，log域

    def forward(self, audio_embed, audio_lengths, phone_ids, phone_lengths):
        """
        train、predict、valid的共同部分
        """
        # 分别提取特征
        audio_embed = self.upsample_layer(audio_embed)
        audio_feature = self.audio_network(audio_embed)

        phone_embed = self.phone_embedding(phone_ids).transpose(1, 2)
        phone_feature = self.phone_network(phone_embed)

        # 对音频与音素特征进行匹配，算出音频每一帧与音素的相似度（似然，每一帧的 log P(音频|音素)）
        posterior = torch.einsum(
            "bct,bcl->blt", audio_feature, phone_feature
        ) / math.sqrt(audio_feature.shape[1])
        # 熵不变性操作，保证音素序列长度变化时，熵尽量不变，让音素序列长度不要对置信度产生影响，保证置信度的可参考性
        # from: https://kexue.fm/archives/8823
        entropy_invariance_factor = (
            (torch.log(phone_lengths) / self.log_base).unsqueeze(-1).unsqueeze(-1)
        )
        posterior = posterior * entropy_invariance_factor

        # 没有音频信息时，每一帧的先验概率 log P(音素)
        phone_prior = (
            self.phone_prior(phone_ids) * entropy_invariance_factor
        )  # 也要熵不变性
        alignment_prior = generate_prior(
            posterior.shape,
            audio_lengths * self.upsample_rate,
            phone_lengths,
            device=posterior.device,
        )  # 不需要熵不变性，因为 matrix_prior 已经熵不变了

        # 结合 P(音频|音素) 与 P(音素) 求出 P(音素|音频)
        logits = posterior + alignment_prior + phone_prior
        logits = F.pad(
            logits, (0, 0, 1, 0), mode="constant", value=-2
        )  # dummy class，让无处安放的音素都归到 dummy class 中
        align_matrix = F.log_softmax(
            logits,
            dim=1,
        )
        align_matrix = align_matrix[:, 1:, :]  # remove dummy class

        return align_matrix

    def _forward_sum_loss(self, align_matrix, audio_lengths, phone_lengths):
        # add untrainable blank class
        log_probs = F.pad(align_matrix, (0, 0, 1, 0), mode="constant", value=-1)
        log_probs = rearrange(log_probs, "b l t -> t b l")

        # targets
        # ctc_loss 自带 target_lengths 参数，不需要再置零
        targets = torch.arange(1, log_probs.shape[-1] + 1, device=log_probs.device)
        targets = repeat(targets, "l -> b l", b=log_probs.shape[1])

        return F.ctc_loss(
            log_probs,
            targets,
            audio_lengths * self.upsample_rate,
            phone_lengths,
            reduction="sum",
        )

    def _pseudo_label_loss(
        self, align_matrix, audio_lengths, phone_lengths, return_pseudo_label=False
    ):
        result = decode_matrix(
            align_matrix,
            audio_lengths * self.upsample_rate,
            phone_lengths,
            return_confidence=False,
        )
        indices = repeat(
            torch.arange(0, align_matrix.shape[1], device=align_matrix.device),
            "l -> b l",
            b=align_matrix.shape[0],
        )
        pseudo_label = generate_matrix(indices, result, align_matrix.shape)

        frame_confidence = (align_matrix * pseudo_label).sum(1)
        mask = frame_confidence > self.pseudo_label_threshold
        loss = -(frame_confidence * mask).sum()

        mask = frame_confidence > -14
        mean_confidence = (frame_confidence * mask).sum() / (mask.sum() + 1e-6)
        self.pseudo_label_threshold = (
            0.99 * self.pseudo_label_threshold + (1 - 0.99) * mean_confidence
        )  # 自适应阈值
        if return_pseudo_label:
            return loss, pseudo_label
        else:
            return loss

    def _classification_loss(self, align_matrix, phone_intervals):
        # TODO: 类别不平衡
        indices = repeat(
            torch.arange(0, align_matrix.shape[1], device=align_matrix.device),
            "l -> b l",
            b=align_matrix.shape[0],
        )
        label = generate_matrix(indices, phone_intervals, align_matrix.shape)

        loss = -(align_matrix * label).sum()

        return loss

    def _get_losses(
        self,
        label_types,
        audio_embed,
        audio_lengths,
        phone_ids,
        phone_lengths,
        phone_intervals,
        **kwargs
    ):
        weak_label = label_types == 1
        full_label = label_types == 0

        align_matrix = self.forward(
            audio_embed, audio_lengths, phone_ids, phone_lengths
        )

        # 兼容 no_label 的loss不在这个task里

        loss_dict = {
            "forced_alignment/forward_sum_loss": self._forward_sum_loss(
                align_matrix[weak_label],
                audio_lengths[weak_label],
                phone_lengths[weak_label],
            ),
        }
        if weak_label.any():
            loss_dict.update(
                {
                    "forced_alignment/pseudo_label_loss": self._pseudo_label_loss(
                        align_matrix[weak_label],
                        audio_lengths[weak_label],
                        phone_lengths[weak_label],
                    )
                }
            )
        if full_label.any():
            loss_dict.update(
                {
                    "forced_alignment/classification_loss": self._classification_loss(
                        align_matrix[full_label], phone_intervals[full_label]
                    )
                }
            )

        return loss_dict

    def train(self, *args, **kwargs):
        """
        获得loss_dict
        """
        return self._get_losses(*args, **kwargs)

    def valid(self, **kwargs):
        """
        获得metric_dict和figure
        """
        # metric_dict = {}
        # taichi解码
        # 绘图
        # figure = plt.
        pass

    def predict(self):
        """
        获得result_dict
        """
        pass


if __name__ == "__main__":

    def test_alignment_head():
        device = torch.device("cuda")

        scale_factor = 8
        alignment = ForcedAlignmentTask(
            scale_factor,
            nn.Conv1d(512, 512, 3, padding="same"),
            100,
            512,
            nn.Conv1d(512, 512, 3, padding="same"),
        ).to(device)
        B, C, T, L = 4, 512, 1000, 100
        label_types = torch.randint(0, 2, (B,), device=device)
        audio_embed = torch.randn(B, C, T, device=device)
        audio_lengths = torch.randint(int(T // 2), T, (B,), device=device)
        phone_ids = torch.randint(0, L, (B, L), device=device)
        phone_lengths = torch.randint(int(L // 2), L, (B,), device=device)

        print("label_types", label_types)

        # forward
        align_matrix = alignment(audio_embed, audio_lengths, phone_ids, phone_lengths)
        phone_intervals = decode_matrix(
            align_matrix,
            audio_lengths * scale_factor,
            phone_lengths,
            return_confidence=False,
        )

        import matplotlib.pyplot as plt

        print("align_matrix.shape", align_matrix.shape)
        plt.imshow(
            align_matrix[0].exp().detach().cpu(),
            origin="lower",
            aspect="auto",
            vmin=0,
            vmax=1,
        )
        plt.colorbar()
        plt.savefig("1.png")

        # forward sum loss
        loss = alignment._forward_sum_loss(align_matrix, audio_lengths, phone_lengths)
        print("forward_sum_loss", loss)

        # pseudo_label_loss
        loss, pseudo_label = alignment._pseudo_label_loss(
            align_matrix, audio_lengths, phone_lengths, return_pseudo_label=True
        )
        print("pseudo_label_loss", loss)
        import matplotlib.pyplot as plt

        plt.imshow(
            pseudo_label[0].cpu(),
            vmin=0,
            vmax=1,
            cmap="gray",
            origin="lower",
            aspect="auto",
        )

        plt.savefig("2.png")

        # classification loss
        loss = alignment._classification_loss(align_matrix, phone_intervals)
        print("classification_loss", loss)

        # loss dict
        loss = alignment.train(
            label_types,
            audio_embed,
            audio_lengths,
            phone_ids,
            phone_lengths,
            phone_intervals,
        )
        print(loss)

        sum(loss.values()).backward()

    test_alignment_head()
