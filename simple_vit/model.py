import torch
import torch.nn as nn


class SimpleViT(nn.Module):

    def __init__(
        self,
        image_size=32,
        patch_size=4,
        num_classes=10,
        embed_dim=128,
        num_heads=4,
        num_layers=6
    ):

        super().__init__()


        # =========================
        # 1. Patch Embedding
        # =========================

        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )


        # CIFAR-10:
        # 32x32 image
        # patch size = 4x4
        #
        # 32 / 4 = 8
        # 8 x 8 = 64 patches

        num_patches = (
            image_size // patch_size
        ) ** 2


        # =========================
        # 2. CLS token
        # =========================

        self.cls_token = nn.Parameter(
            torch.zeros(
                1,
                1,
                embed_dim
            )
        )


        # =========================
        # 3. Position Embedding
        # =========================

        self.pos_embedding = nn.Parameter(
            torch.zeros(
                1,
                num_patches + 1,
                embed_dim
            )
        )


        # =========================
        # 4. Transformer Encoder
        # =========================

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True
        )


        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )


        # =========================
        # 5. Classification Head
        # =========================

        self.classifier = nn.Linear(
            embed_dim,
            num_classes
        )


    def forward(self, x):

        # ---------------------------------
        # x:
        # [B, 3, 32, 32]
        # ---------------------------------


        # =========================
        # Patch Embedding
        # =========================

        x = self.patch_embed(x)

        # [B, 128, 8, 8]


        x = x.flatten(2)

        # [B, 128, 64]


        x = x.transpose(1, 2)

        # [B, 64, 128]


        # =========================
        # CLS token
        # =========================

        batch_size = x.shape[0]


        cls_tokens = self.cls_token.expand(
            batch_size,
            -1,
            -1
        )

        # [B, 1, 128]


        x = torch.cat(
            [cls_tokens, x],
            dim=1
        )

        # [B, 65, 128]


        # =========================
        # Position Embedding
        # =========================

        x = x + self.pos_embedding


        # =========================
        # Transformer
        # =========================

        x = self.transformer(x)

        # [B, 65, 128]


        # =========================
        # CLS token output
        # =========================

        cls_output = x[:, 0]

        # [B, 128]


        # =========================
        # Classification
        # =========================

        output = self.classifier(
            cls_output
        )

        # [B, 10]


        return output