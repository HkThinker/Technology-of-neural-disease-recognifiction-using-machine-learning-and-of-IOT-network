"""
展示数据的样貌
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import pywt
import soundfile as sf

# 读取wav文件
audio_data, sample_rate = sf.read("F:/工程/Python/Python办公室自动化/PythonApplication1/Machine learning/Dataset/diagnosis/train/audio/ad/adrso024.wav")

# 定义窗口大小和重叠率
window_size = 102400
overlap = 0.5

# 计算每个窗口的开始和结束位置
hop_length = int(window_size * (1 - overlap))
start = window_size // 2
stop = len(audio_data) - window_size // 2

# 取第二个窗口的数据
first_frame = audio_data[start+hop_length*200: start+hop_length*200+window_size]
time = np.linspace(0, len(first_frame) / sample_rate, len(first_frame))
# 绘制信号波形图
plt.subplot(2, 2, 1)
plt.plot(time,first_frame)
plt.title("Waveform_Alzheimer's disease(1)")
plt.ylabel('Amplitude')
plt.xlabel('Time (sec)')
# 计算频域图
freq, power = signal.welch(first_frame, fs=sample_rate, window='hann', nperseg=window_size, noverlap=window_size*overlap)

# 绘制频域图
plt.subplot(2,2, 2)
plt.semilogy(freq, power)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.title("Power Spectrum_Alzheimer's disease(1)")


# 取第二个窗口的数据
second_frame = audio_data[start+hop_length*100: start+hop_length*100+window_size]
time = np.linspace(0, len(second_frame) / sample_rate, len(second_frame))
# 绘制信号波形图
plt.subplot(2, 2, 3)
plt.plot(time,second_frame)
plt.title("Waveform_Alzheimer's disease(2)")
plt.ylabel('Amplitude')
plt.xlabel('Time (sec)')
# 计算频域图
freq, power = signal.welch(second_frame, fs=sample_rate, window='hann', nperseg=window_size, noverlap=window_size*overlap)

# 绘制频域图
plt.subplot(2,2, 4)
plt.semilogy(freq, power)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.title("Power Spectrum_Alzheimer's disease(2)")
plt.tight_layout()
plt.show()

#"""
#数据提取特征个体代码
#"""
#import opensmile
#import pandas as pd
#import numpy as np
#import os
#from scipy.io import wavfile

#print(os.getcwd())  # 输出当前工作目录
## 定义openSMILE配置文件路径
#config_file = 'F:/工程/Python/Python办公室自动化/PythonApplication1/Machine learning/Dataset/diagnosis/my_config.conf'

## 定义音频文件路径
#audio_file = 'adrso024.wav'

## 读取音频文件
#sampling_rate, signal = wavfile.read(audio_file)

## 创建openSMILE实例
## eGeMAPSv02 具有88个特征
#smile = opensmile.Smile(
#    feature_set=opensmile.FeatureSet.eGeMAPSv02,
#    feature_level=opensmile.FeatureLevel.Functionals,
#)

#print(smile.feature_names)
#print(len(smile.feature_names))
## 提取特征
#features = smile.process_signal(signal, sampling_rate)

## 将特征数据类型转换为float
#features = np.array(features).astype(float)

## 将特征转换为pandas DataFrame
#df = pd.DataFrame(features, index=[0])

## 将DataFrame保存为CSV文件
#df.to_csv('features.csv', index=False)

#"""
#对数据清洗,删除数据的无效数据
#"""
#import os
#from pydub import AudioSegment

#metadata_file = "metadata.csv"
#output_dir = "output"

#if not os.path.exists(output_dir):
#    os.makedirs(output_dir)

#with open(metadata_file, "r") as f:
#    lines = f.readlines()[1:]  # 忽略文件头
#    for line in lines:
#        line = line.strip().split(",")
#        speaker = line[1]
#        begin = int(line[2])
#        end = int(line[3])

#        if speaker == "INV":
#            continue

#        audio_file = os.path.join("data", line[0] + ".wav")
#        audio = AudioSegment.from_wav(audio_file)
#        audio = audio[begin:end]
#        audio.export(os.path.join(output_dir, line[0] + ".wav"), format="wav")





