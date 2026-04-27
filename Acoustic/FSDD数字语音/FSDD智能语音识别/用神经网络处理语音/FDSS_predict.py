import torch
import torchaudio
import os
from FDSS_model_def import AudioCNN


MODEL_PATH = 'fsdd_model.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        print(f"错误：找不到模型文件 {MODEL_PATH}，请先运行 train.py！")
        return None
    #实例化结构
    model = AudioCNN()

    #加载参数
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    #转移到设备并开启评估模式
    model.to(DEVICE)
    model.eval()
    return model

def predict_single_file(model, file_path):
    #预处理
    target_fs = 8000
    max_len = 8000

    try:
        waveform, fs = torchaudio.load(file_path)
    except Exception as e:
        print(f"读取失败: {e}")
        return

    #如果通道数大于 1 (比如立体声是 2)，我们取平均值变成 1
    if waveform.shape[0] > 1:
        print(f"检测到多声道音频 {waveform.shape}，正在合并为单声道...")
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    #重采样
    if fs != target_fs:
        waveform = torchaudio.transforms.Resample(fs, target_fs)(waveform)
    #填充/截断
    if waveform.shape[1] < max_len:
        padding = torch.zeros(1, max_len - waveform.shape[1])
        waveform = torch.cat((waveform, padding), dim=1)
    else:
        waveform = waveform[:, :max_len]
    #转MFCC
    mfcc = torchaudio.transforms.MFCC(
        sample_rate=target_fs, n_mfcc=13,
        melkwargs={'n_fft': 400, 'hop_length': 160, 'n_mels': 23, 'center': False}
    )(waveform)

    #预测
    input_tensor = mfcc.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(input_tensor)
        pred = logits.argmax(1).item()

    return pred


if __name__ == '__main__':
    #加载模型
    print(f"正在加载模型... (使用设备: {DEVICE})")
    model = load_trained_model()

    if model:
        #指定文件
        test_file = r'/Acoustic/FSDD数字语音/dataset\0to9wav\0_myvoice_1.wav'

        print(f"正在预测: {test_file}")
        if os.path.exists(test_file):
            result = predict_single_file(model, test_file)
            print(f"-----------------------")
            print(f"预测结果: 数字 {result}")
            print(f"-----------------------")
        else:
            print("找不到测试文件，请修改 test_file 路径")