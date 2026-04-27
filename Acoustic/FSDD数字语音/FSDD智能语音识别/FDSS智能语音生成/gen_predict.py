import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import os
from gen_model import Generator

MODEL_PATH = 'fsdd_gan_G.pth'
OUTPUT_DIR = 'generated_audio'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_digit(digit, model):
    model.eval()

    #制造随机噪声 + 指定标签
    noise = torch.randn(1, 100, 1, 1).to(DEVICE)
    label = torch.tensor([digit]).to(DEVICE)

    #生成频谱图 (值在 -1 到 1 之间)
    with torch.no_grad():
        spec_img = model(noise, label)

    #反归一化 (变回分贝值)
    #先 squeeze 掉 batch 和 channel 维度，变成 [64, 64]
    spec_img = spec_img.squeeze(0).squeeze(0)

    spec_img = (spec_img + 1) / 2  # 变回 0-1
    spec_img = spec_img * 100 - 50  # 变回分贝 (近似范围)

    #从分贝转回振幅 (使用数学公式替代 DBToAmplitude)
    spec_img = torch.pow(10, spec_img / 20)

    #确保 spec_img 在 GPU 上，并且增加 Batch 和 Channel 维度以便后续处理
    #InverseMelScale 需要 [..., n_mels, time]
    spec_img = spec_img.unsqueeze(0).to(DEVICE)

    inverse_mel = torchaudio.transforms.InverseMelScale(
        n_stft=201,
        n_mels=64,
        sample_rate=8000
    ).to(DEVICE)

    #执行变换
    spec_img = inverse_mel(spec_img)

    griffin_lim = torchaudio.transforms.GriffinLim(
        n_fft=400,
        n_iter=32,
        hop_length=126,
        power=1
    ).to(DEVICE)

    waveform = griffin_lim(spec_img)

    return waveform.cpu(), spec_img.cpu()


if __name__ == '__main__':
    #加载模型
    if not os.path.exists(MODEL_PATH):
        print("请先运行 gen_train.py！")
        exit()

    print(f"正在加载模型... 设备: {DEVICE}")
    generator = Generator().to(DEVICE)

    #加上 weights_only=True 防止警告，如果报错则去掉该参数
    try:
        generator.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    except:
        generator.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    #输入你想生成的数字
    target_digit = 9

    print(f"正在生成数字 {target_digit} 的语音...")
    try:
        wav, spec = generate_digit(target_digit, generator)

        #保存音频
        filename = f"{OUTPUT_DIR}/{target_digit}_generated.wav"
        #wav 现在的形状可能是 [1, 8000] 或者 [8000]，save 需要 [Channels, Time]
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        torchaudio.save(filename, wav, 8000)
        print(f"音频已保存: {filename}")

        plt.figure()
        #spec 是 [1, 201, 64] 或者 [201, 64]，去掉 batch 维画图
        if spec.dim() == 3:
            spec = spec.squeeze(0)

        plt.imshow(spec.numpy(), origin='lower', aspect='auto')
        plt.title(f"Generated Spectrogram for '{target_digit}'")
        plt.show()

    except Exception as e:
        print(f"生成失败: {e}")
        import traceback

        traceback.print_exc()