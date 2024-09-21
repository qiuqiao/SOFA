import math

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

# from ..network.layer.conv_stft import ConvMelSpectrogram
from ..utils.alignment import decode_matrix, generate_matrix, generate_prior

# class Encoder(nn.Module):
#     def __init__(
#         self,
#         network,
#         mel_args: dict,
#     ):
#         super().__init__()
#         self.stft = ConvMelSpectrogram(**mel_args)
#         self.network = network

#     def forward(self, wav):
#         return self.network(self.stft(wav))

#     # def ssl_loss(self)


class AlignerHead(nn.Module):
    """
    AlignerHead负责对给定正确的普通音素序列进行对齐，不负责音素序列有误/缺少音素序列/SP等特殊因素的情况，
    但在不影响主要任务的情况下，可以尽量增强鲁棒性。
    """

    def __init__(
        self,
        scale_factor,
        head_network,
        num_phones,
        phone_embedding_dims,
        phone_encoder_network,
        ema_factor=0.99,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.aligner_upsample_layer = nn.Upsample(scale_factor=scale_factor)
        self.head_network = head_network

        self.phone_embedding = nn.Embedding(num_phones, phone_embedding_dims)
        self.phone_encoder_network = phone_encoder_network

        self.register_buffer("log_base", torch.log(torch.tensor(25)))
        self.register_buffer("pseudo_label_threshold", torch.tensor([-2]))
        self.ema_factor = ema_factor

    def forward(self, audio_embed, audio_lengths, phone_ids, phone_lengths):
        audio_embed = self.head_network(self.aligner_upsample_layer(audio_embed))
        phone_embed = self.phone_encoder_network(
            self.phone_embedding(phone_ids).transpose(1, 2)
        )

        posterior = torch.einsum("bct,bcl->blt", audio_embed, phone_embed) / math.sqrt(
            audio_embed.shape[1]
        )
        prior = generate_prior(
            posterior.shape,
            audio_lengths * self.scale_factor,
            phone_lengths,
            device=posterior.device,
        )
        entropy_invariance_factor = (  # from: https://kexue.fm/archives/8823
            (torch.log(phone_lengths) / self.log_base).unsqueeze(-1).unsqueeze(-1)
        )
        logits = entropy_invariance_factor * posterior + prior
        logits = torch.cat(  # dummy class
            (
                torch.full(
                    (audio_embed.shape[0], 1, audio_embed.shape[2]),
                    -2,
                    device=logits.device,
                ),
                logits,
            ),
            dim=1,
        )
        align_matrix = F.log_softmax(
            logits,
            dim=1,
        )

        return align_matrix[:, 1:, :]

    def forward_sum_loss(self, align_matrix, audio_lengths, phone_lengths):
        # add untrainable blank class
        log_probs = torch.cat(
            (
                torch.full(
                    (align_matrix.shape[0], 1, align_matrix.shape[2]),
                    -1,
                    device=align_matrix.device,
                ),
                align_matrix,
            ),
            dim=1,
        )
        log_probs = rearrange(log_probs, "b l t -> t b l")

        # targets
        targets = torch.arange(1, log_probs.shape[-1] + 1, device=log_probs.device)
        targets = repeat(targets, "l -> b l", b=log_probs.shape[1])

        return F.ctc_loss(
            log_probs,
            targets,
            audio_lengths * self.scale_factor,
            phone_lengths,
            reduction="sum",
        )

    def pseudo_label_loss(self, align_matrix, audio_lengths, phone_lengths):
        result = decode_matrix(
            align_matrix,
            audio_lengths * self.scale_factor,
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
            self.ema_factor * self.pseudo_label_threshold
            + (1 - self.ema_factor) * mean_confidence
        )  # 自适应阈值

        return loss, pseudo_label

    def classification_loss(
        self, align_matrix, audio_lengths, phone_lengths, ph_id_seq, ph_id_intervals
    ):
        # TODO: 类别不平衡
        indices = repeat(
            torch.arange(0, align_matrix.shape[1], device=align_matrix.device),
            "l -> b l",
            b=align_matrix.shape[0],
        )
        label = generate_matrix(indices, ph_id_intervals, align_matrix.shape)

        loss = -(align_matrix * label).sum()

        return loss

    # def training_step


if __name__ == "__main__":

    def test_aligner_head():
        device = torch.device("cuda")

        aligner = AlignerHead(
            8,
            nn.Conv1d(512, 512, 3, padding="same"),
            100,
            512,
            nn.Conv1d(512, 512, 3, padding="same"),
        ).to(device)
        B, C, T, L = 4, 512, 1000, 100
        audio_embed = torch.randn(B, C, T, device=device)
        audio_lengths = torch.randint(int(T // 2), T, (B,), device=device)
        phone_ids = torch.randint(0, L, (B, L), device=device)
        phone_lengths = torch.randint(int(L // 2), L, (B,), device=device)

        # forward
        align_matrix = aligner(audio_embed, audio_lengths, phone_ids, phone_lengths)

        # import matplotlib.pyplot as plt

        # print(align_matrix.shape)
        # plt.imshow(
        #     align_matrix[0].exp().detach().cpu(),
        #     origin="lower",
        #     aspect="auto",
        #     vmin=0,
        #     vmax=1,
        # )
        # plt.colorbar()
        # plt.show()

        # # forward sum loss

        # loss = aligner.forward_sum_loss(align_matrix, audio_lengths, phone_lengths)
        # print(loss)

        # pseudo_label_loss
        loss, pseudo_label = aligner.pseudo_label_loss(
            align_matrix, audio_lengths, phone_lengths
        )
        print(loss)
        import matplotlib.pyplot as plt

        plt.imshow(
            pseudo_label[0].cpu(),
            vmin=0,
            vmax=1,
            cmap="gray",
            origin="lower",
            aspect="auto",
        )

        plt.show()

    test_aligner_head()


# class SOFA(L.LightningModule):
#     def __init__(
#         self,
#         # encoder:
#         audio_hoplength,
#         audio_encoder,
#         # aligner:
#         aligner_hoplength,
#         aligner_head,
#         phone_encoder,
#     ):
#         super().__init__()
#         assert (
#             audio_hoplength % aligner_hoplength == 0
#         ), f"audio_hoplength ({audio_hoplength}) should be divisible by aligner_hoplength ({aligner_hoplength})"

#         # encoder
#         self.stft = ConvSTFT(audio_hoplength * 4, audio_hoplength, trainable=True)
#         self.audio_encoder = audio_encoder

#         # aligner head
#         self.aligner_upsample_layer = nn.Upsample(
#             scale_factor=int(audio_hoplength // aligner_hoplength)
#         )
#         self.aligner_head = aligner_head
#         self.phone_encoder = phone_encoder  # 包含了Embedding层

#     # def forward(self, )

#     def training_step(self, batch, batch_idx):
#         (
#             wav,
#             lengths,
#             normal_id_seq,
#             normal_id_lengths,
#             normal_interval_seq,
#             special_id_seq,
#             special_interval_seq,
#             label_type,
#         ) = batch

#         weak_label = label_type == 1
#         full_label = label_type == 0

#         loss_dict = {}

#         # TODO: no label
#         audio_embed = self.audio_encoder(self.stft(wav))

#         # weak label
#         if (weak_label).any():
#             audio_embed = audio_embed[weak_label]
#             phone_embed = self.phone_encoder(normal_id_seq[weak_label])

#             # task: aligner
#             if self.aligner_task_enabled:
#                 audio_embed_aligner = self.aligner_head(audio_embed)
#                 # 包含了上采样到aligner_hoplength
#                 align_matrix = torch.einsum(
#                     "bct,bcl->btl", audio_embed_aligner, phone_embed
#                 )
#                 # TODO: prior
#                 # forward sum loss
#                 B, T, L = align_matrix.shape
#                 align_matrix = torch.cat(
#                     [torch.full((B, T, 1), -1), align_matrix], dim=2
#                 )
#                 ## blank label
#                 log_probs = F.log_softmax(align_matrix, dim=2)
#                 log_probs = rearrange(log_probs, "b t l -> t b l")
#                 forward_sum_loss = F.ctc_loss(
#                     log_probs,
#                     targets=torch.range(1, align_matrix.shape[2]),
#                     input_lengths=int(
#                         lengths[weak_label] // self.aligner_hoplength + 0.5
#                     ),
#                     target_lengths=normal_id_lengths[weak_label],
#                     reduction="sum",
#                 )
#                 loss_dict.update({"forward_sum_loss": forward_sum_loss})
#                 # TODO:pseudo label loss

#                 # TODO: MLM loss

#             # task: ASR

#         # TODO: full label
