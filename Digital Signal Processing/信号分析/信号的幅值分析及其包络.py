import librosa
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

#加载信号
wave_path = Path(__file__).resolve().parents[2] / 'Acoustic' / '声信息特征提取' / 'audios' / 'audio_raw_vad.wav'
waveform,sample_rate = librosa.load(wave_path,sr=None)
print(waveform)


#定义一个AE函数 取每一帧振幅最大值为包络
def ae_fuction(waveform,frame_length,hop_length):
    if len(waveform) % hop_length != 0:
        frame_num = int((len(waveform) - frame_length) / hop_length) + 1
        pad_num = frame_num * hop_length + frame_length - len(waveform)
        waveform = np.pad(waveform, (0, pad_num), 'wrap')
    frame_num = int((len(waveform) - frame_length) / hop_length) + 1
    waveform_ae_max = []
    waveform_ae_min = []
    for i in range(frame_num):
        current_frame = waveform[i*hop_length:(i+1)*hop_length+frame_length]
        waveform_ae_max.append(max(current_frame))
        waveform_ae_min.append(min(current_frame))
    return waveform_ae_max,waveform_ae_min

#设置参数调用帧长
frame_length = 80
hop_length = int(frame_length/2)
waveform_AE = ae_fuction(waveform,frame_length,hop_length)

#绘制
frame_scale = np.arange(0,len(waveform_AE[0]))
time_scale = librosa.frames_to_time(frame_scale,hop_length=hop_length)
plt.figure(figsize=(20,10))
plt.plot(time_scale,waveform_AE[0],color='r')
plt.plot(time_scale,waveform_AE[1],color='r')
librosa.display.waveshow(waveform)
plt.title('AE of audio')
plt.show()
