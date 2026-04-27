import os
import torch
import torchaudio
import joblib
from 随机森林model import AudioFeatureExtractor


MODEL_PATH = 'fsdd_rf_model.pkl'


def predict_single_file(model, file_path, extractor):
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return

    try:
        waveform, sample_rate = torchaudio.load(file_path)
    except Exception as e:
        print(f"读取失败: {e}")
        return

    #提取特征 (得到26维向量)
    feature_vector = extractor.process_waveform(waveform, sample_rate)

    #调整形状
    feature_vector = feature_vector.reshape(1, -1)

    #预测
    pred = model.predict(feature_vector)[0]
    #每个数字的概率
    probs = model.predict_proba(feature_vector)[0]
    confidence = probs[pred]

    return pred, confidence


if __name__ == '__main__':
    #加载模型
    if not os.path.exists(MODEL_PATH):
        print("没找到模型文件，请先运行 train.py")
        exit()

    print("正在加载随机森林模型...")
    model = joblib.load(MODEL_PATH)
    extractor = AudioFeatureExtractor()

    #指定文件
    test_file = r'/Acoustic/FSDD数字语音/dataset\test\4_george_0.wav'

    #预测
    print(f"正在预测: {test_file}")
    result, conf = predict_single_file(model, test_file, extractor)

    print("-" * 30)
    print(f"预测结果: 数字 {result}")
    print(f"置信度: {conf * 100:.2f}%")
    print("-" * 30)