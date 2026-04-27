import torch
import torchaudio
import numpy as np


class AudioFeatureExtractor:
    def __init__(self, target_sample_rate=8000):
        self.target_sample_rate = target_sample_rate
        #定义MFCC变换器
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=target_sample_rate,
            n_mfcc=13,
            melkwargs={'n_fft': 400, 'hop_length': 160, 'n_mels': 23, 'center': False}
        )

    def process_waveform(self, waveform, sample_rate):
        #转单声道
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        #重采样
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
            waveform = resampler(waveform)
        #归一化 (抗音量差异)
        if torch.max(torch.abs(waveform)) > 0:
            waveform = waveform / torch.max(torch.abs(waveform))
        #提取 MFCC [1, 13, Time]
        mfcc = self.mfcc_transform(waveform)


        mfcc_mean = torch.mean(mfcc, dim=2).squeeze().numpy()
        mfcc_std = torch.std(mfcc, dim=2).squeeze().numpy()

        #拼起来
        features = np.concatenate([mfcc_mean, mfcc_std])
        return features