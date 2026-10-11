import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Transformer Stage
# ============================================================

class TransformerStage(nn.Module):

    def __init__(
        self,
        in_channels,
        embed_dim,
        kernel_size,
        stride,
        padding,
        num_heads,
        num_layers=1
    ):

        super().__init__()


        # =========================
        # Overlapping Patch Embedding
        # =========================

        self.patch_embed = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )


        # =========================
        # Transformer Encoder
        # =========================

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )


        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )


    def forward(self, x):

        # --------------------------------
        # Patch Embedding
        # --------------------------------

        x = self.patch_embed(x)

        # example:
        # [B,32,32,32]


        B, C, H, W = x.shape


        # --------------------------------
        # Feature Map -> Tokens
        # --------------------------------

        x = x.flatten(2)

        # [B,C,H*W]


        x = x.transpose(1, 2)

        # [B,H*W,C]


        # --------------------------------
        # Transformer
        # --------------------------------

        x = self.transformer(x)


        # --------------------------------
        # Tokens -> Feature Map
        # --------------------------------

        x = x.transpose(1, 2)

        x = x.reshape(
            B,
            C,
            H,
            W
        )


        return x


# ============================================================
# Simple SegFormer
# ============================================================

class SimpleSegFormer(nn.Module):

    def __init__(
        self,
        num_classes=3
    ):

        super().__init__()


        # ====================================================
        # Hierarchical Transformer Encoder
        # ====================================================

        # 128x128
        #   ↓
        # 32x32

        self.stage1 = TransformerStage(
            in_channels=3,
            embed_dim=32,
            kernel_size=7,
            stride=4,
            padding=3,
            num_heads=4
        )


        # 32x32
        #   ↓
        # 16x16

        self.stage2 = TransformerStage(
            in_channels=32,
            embed_dim=64,
            kernel_size=3,
            stride=2,
            padding=1,
            num_heads=4
        )


        # 16x16
        #   ↓
        # 8x8

        self.stage3 = TransformerStage(
            in_channels=64,
            embed_dim=128,
            kernel_size=3,
            stride=2,
            padding=1,
            num_heads=8,

        )


        # 8x8
        #  ↓
        # 4x4

        self.stage4 = TransformerStage(
            in_channels=128,
            embed_dim=256,
            kernel_size=3,
            stride=2,
            padding=1,
            num_heads=8
        )


        # ====================================================
        # Lightweight Decoder
        # ====================================================

        decoder_dim = 64


        self.linear1 = nn.Conv2d(
            32,
            decoder_dim,
            kernel_size=1
        )

        self.linear2 = nn.Conv2d(
            64,
            decoder_dim,
            kernel_size=1
        )

        self.linear3 = nn.Conv2d(
            128,
            decoder_dim,
            kernel_size=1
        )

        self.linear4 = nn.Conv2d(
            256,
            decoder_dim,
            kernel_size=1
        )


        # Fuse four feature maps

        self.fuse = nn.Sequential(

            nn.Conv2d(
                decoder_dim * 4,
                decoder_dim,
                kernel_size=1
            ),

            nn.BatchNorm2d(
                decoder_dim
            ),

            nn.ReLU(
                inplace=True
            )
        )


        # Segmentation Head

        self.classifier = nn.Conv2d(
            decoder_dim,
            num_classes,
            kernel_size=1
        )


    def forward(self, x):

        input_size = x.shape[2:]


        # ====================================================
        # Transformer Encoder
        # ====================================================

        f1 = self.stage1(x)

        # [B,32,32,32]


        f2 = self.stage2(f1)

        # [B,64,16,16]


        f3 = self.stage3(f2)

        # [B,128,8,8]


        f4 = self.stage4(f3)

        # [B,256,4,4]


        # ====================================================
        # Decoder
        # ====================================================

        f1 = self.linear1(f1)

        f2 = self.linear2(f2)

        f3 = self.linear3(f3)

        f4 = self.linear4(f4)


        # --------------------------------
        # 全部恢复到 Stage 1 的尺寸
        # --------------------------------

        target_size = f1.shape[2:]


        f2 = F.interpolate(
            f2,
            size=target_size,
            mode="bilinear",
            align_corners=False
        )


        f3 = F.interpolate(
            f3,
            size=target_size,
            mode="bilinear",
            align_corners=False
        )


        f4 = F.interpolate(
            f4,
            size=target_size,
            mode="bilinear",
            align_corners=False
        )


        # ====================================================
        # Feature Fusion
        # ====================================================

        x = torch.cat(
            [
                f1,
                f2,
                f3,
                f4
            ],
            dim=1
        )


        x = self.fuse(x)


        # ====================================================
        # Segmentation
        # ====================================================

        x = self.classifier(x)


        # [B,3,32,32]
        #
        # 恢复原图尺寸
        # [B,3,128,128]

        x = F.interpolate(
            x,
            size=input_size,
            mode="bilinear",
            align_corners=False
        )


        return x