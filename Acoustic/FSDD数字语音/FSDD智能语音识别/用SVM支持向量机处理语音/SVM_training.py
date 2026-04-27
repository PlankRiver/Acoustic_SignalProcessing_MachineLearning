import os
import torch
import torchaudio
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from SVM_model import AudioFeatureExtractor


BASE_DIR = r'/Acoustic/FSDD数字语音/dataset'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
TEST_DIR = os.path.join(BASE_DIR, 'test')
MODEL_SAVE_PATH = 'fsdd_svm_model.pkl'


def load_data(data_dir, extractor):
    X = []
    y = []
    files = [f for f in os.listdir(data_dir) if f.endswith('.wav')]

    print(f"正在处理 {data_dir} ...")
    for file_name in files:
        file_path = os.path.join(data_dir, file_name)
        try:
            if '_' in file_name:
                label = int(file_name.split('_')[0])
            else:
                label = int(file_name[0])
        except:
            continue

        try:
            waveform, sample_rate = torchaudio.load(file_path)
        except:
            continue

        feature_vector = extractor.process_waveform(waveform, sample_rate)
        X.append(feature_vector)
        y.append(label)

    return np.array(X), np.array(y)



if __name__ == '__main__':
    extractor = AudioFeatureExtractor()

    print("正在读取训练集...")
    X_train, y_train = load_data(TRAIN_DIR, extractor)

    print("正在读取测试集...")
    X_test, y_test = load_data(TEST_DIR, extractor)

    print(f"训练集形状: {X_train.shape}")

    #数据标准化
    clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True))

    print("开始训练 SVM...")
    clf.fit(X_train, y_train)

    #验证
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"训练集准确率: {train_acc * 100:.2f}%")
    print(f"测试集准确率: {test_acc * 100:.2f}%")

    #保存
    joblib.dump(clf, MODEL_SAVE_PATH)
    print(f"模型已保存为: {os.path.abspath(MODEL_SAVE_PATH)}")