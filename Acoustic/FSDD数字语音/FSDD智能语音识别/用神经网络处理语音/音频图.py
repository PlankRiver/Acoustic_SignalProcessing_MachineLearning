import torch
import torchaudio
import matplotlib.pyplot as plt
import os


FSDD_FILE = r'/Acoustic/dataset/test/0_jackson_0.wav'
MY_FILE = r'/Acoustic/FSDD数字语音/dataset\0to9wav\0.wav'


def get_mfcc(path):
    waveform, fs = torchaudio.load(path)
    #转单声道
    if waveform.shape[0] > 1: waveform = torch.mean(waveform, dim=0, keepdim=True)
    #重采样
    if fs != 8000: waveform = torchaudio.transforms.Resample(fs, 8000)(waveform)
    #归一化音量
    waveform = waveform / torch.max(torch.abs(waveform))
    #裁剪/填充
    if waveform.shape[1] < 8000:
        padding = torch.zeros(1, 8000 - waveform.shape[1])
        waveform = torch.cat((waveform, padding), dim=1)
    else:
        waveform = waveform[:, :8000]

    mfcc = torchaudio.transforms.MFCC(sample_rate=8000, n_mfcc=13)(waveform)
    return mfcc[0].numpy()


# 绘图对比
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(get_mfcc(FSDD_FILE), cmap='viridis', aspect='auto', origin='lower')
plt.title("FSDD (Training Data)")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(get_mfcc(MY_FILE), cmap='viridis', aspect='auto', origin='lower')
plt.title("My Recording")
plt.colorbar()

plt.show()