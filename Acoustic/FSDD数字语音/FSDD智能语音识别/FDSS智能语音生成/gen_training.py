import os
import torch
import torch.nn as nn
import torchaudio
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from gen_model import Generator, Discriminator


BASE_DIR = r'/Acoustic/FSDD数字语音/dataset\train'
SAVE_PATH_G = 'fsdd_gan_G.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64
LR = 0.0002
EPOCHS = 100



class SpecDataset(Dataset):
    def __init__(self, data_dir):
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.wav')]
        #定义频谱变换：我们要生成 64x64 的图
        self.spec_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=8000, n_fft=400, n_mels=64, hop_length=126
        ) # hop_length 调成 126 可以在 1秒音频下产生约 64 宽度的图

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        #解析标签
        try:
            label = int(os.path.basename(path).split('_')[0])
        except:
            label = 0

        waveform, sr = torchaudio.load(path)
        #转单声道 & 归一化
        if waveform.shape[0] > 1: waveform = torch.mean(waveform, dim=0, keepdim=True)
        waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-6)

        #统一长度为 1秒 (8000点)
        if waveform.shape[1] < 8000:
            pad = torch.zeros(1, 8000 - waveform.shape[1])
            waveform = torch.cat((waveform, pad), 1)
        else:
            waveform = waveform[:, :8000]

        #转频谱 [1, 64, 64]
        spec = self.spec_transform(waveform)
        spec = torchaudio.transforms.AmplitudeToDB()(spec)  # 转成分贝

        #强制 Resize 到 64x64 (防止计算误差)
        spec = transforms.Resize((64, 64))(spec)

        #归一化到 [-1, 1] (GAN 的标准操作)
        spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-6)
        spec = (spec - 0.5) * 2

        return spec, label



if __name__ == '__main__':
    dataset = SpecDataset(BASE_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    generator = Generator().to(DEVICE)
    discriminator = Discriminator().to(DEVICE)

    criterion = nn.BCELoss()
    opt_g = torch.optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))

    print(f"开始训练 GAN... (设备: {DEVICE})")

    for epoch in range(EPOCHS):
        for i, (imgs, labels) in enumerate(dataloader):
            batch_size = imgs.shape[0]
            real_imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            #定义真假标签
            valid = torch.ones(batch_size).to(DEVICE)
            fake = torch.zeros(batch_size).to(DEVICE)

            opt_d.zero_grad()

            #判别真图
            real_pred = discriminator(real_imgs, labels)
            d_real_loss = criterion(real_pred, valid)

            #判别假图
            noise = torch.randn(batch_size, 100, 1, 1).to(DEVICE)
            gen_imgs = generator(noise, labels)
            fake_pred = discriminator(gen_imgs.detach(), labels)  # detach 防止G的梯度更新
            d_fake_loss = criterion(fake_pred, fake)

            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            opt_d.step()

            opt_g.zero_grad()

            #G 的目标是让 D 以为这是真图 (valid)
            validity = discriminator(gen_imgs, labels)
            g_loss = criterion(validity, valid)

            g_loss.backward()
            opt_g.step()

        print(f"[Epoch {epoch}/{EPOCHS}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

    #保存生成器
    torch.save(generator.state_dict(), SAVE_PATH_G)
    print(f"训练完成！生成器已保存: {SAVE_PATH_G}")