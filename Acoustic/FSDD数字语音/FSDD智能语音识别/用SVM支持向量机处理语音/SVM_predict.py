import os
import torch
import torchaudio
import joblib
from SVM_model import AudioFeatureExtractor


MODEL_PATH = 'fsdd_svm_model.pkl'


def predict_single_file(model, file_path, extractor):
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return None, 0

    try:
        waveform, sample_rate = torchaudio.load(file_path)
    except Exception as e:
        print(f"读取失败: {e}")
        return None, 0
    # 1. 提取特征
    feature_vector = extractor.process_waveform(waveform, sample_rate)
    # 2. 调整形状
    feature_vector = feature_vector.reshape(1, -1)
    # 3. 预测 (StandardScaler 会在模型内部自动运行)
    pred = model.predict(feature_vector)[0]
    probs = model.predict_proba(feature_vector)[0]
    confidence = probs[pred]

    return pred, confidence


if __name__ == '__main__':
    if not os.path.exists(MODEL_PATH):
        print("没找到模型文件，请先运行 train.py")
        exit()

    print("正在加载 SVM 模型...")
    model = joblib.load(MODEL_PATH)
    extractor = AudioFeatureExtractor()

    # 指定文件
    test_file = r'/Acoustic/FSDD数字语音/dataset\0to9wav\9_myvoice_1.wav'

    print(f"正在预测: {test_file}")
    result, conf = predict_single_file(model, test_file, extractor)

    if result is not None:
        print("-" * 30)
        print(f"预测结果: 数字 {result}")
        print(f"置信度: {conf * 100:.2f}%")
        print("-" * 30)