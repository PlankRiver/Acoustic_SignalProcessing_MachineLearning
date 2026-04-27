import torch
import torch.nn as nn

# 设定频谱图的大小 (为了方便卷积，我们强制把频谱图缩放到 64x64)
IMG_SIZE = 64
LATENT_DIM = 100
NUM_CLASSES = 10


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(NUM_CLASSES, LATENT_DIM)

        self.model = nn.Sequential(
            nn.ConvTranspose2d(LATENT_DIM * 2, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_input = self.label_emb(labels).unsqueeze(2).unsqueeze(3)
        combined_input = torch.cat([noise, label_input], 1)
        img = self.model(combined_input)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(NUM_CLASSES, IMG_SIZE * IMG_SIZE)
        self.model = nn.Sequential(
            nn.Conv2d(2, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        # 把标签铺成一张图，和输入图片叠在一起
        label_input = self.label_emb(labels).view(img.size(0), 1, IMG_SIZE, IMG_SIZE)
        combined_input = torch.cat([img, label_input], 1)
        validity = self.model(combined_input)
        return validity.view(-1, 1).squeeze(1)