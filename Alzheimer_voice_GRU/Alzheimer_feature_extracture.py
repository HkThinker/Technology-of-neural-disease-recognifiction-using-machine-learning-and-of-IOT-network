"""
该数据在数据处理上还可以进步,就是删除无效数据
"""
from xml.sax.handler import all_features
import librosa
import os
import opensmile
import pandas as pd
import numpy as np
import os
from scipy.io import wavfile

# 创建openSMILE实例
# eGeMAPSv02 具有88个特征
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

# 定义窗口长度和重叠率
window_length = int(44.1*1000//5) #设置为0.1秒
hop_length = window_length // 2

# 定义文件夹路径
ad_path = "./Machine learning/Dataset/diagnosis/train/seg_audio/ad"


# 创建空数组,能够容纳所有特征
all_features = []

# 遍历文件夹中所有wav文件，并读取数据
for file_name in os.listdir(ad_path):
    if file_name.endswith(".wav"):
        file_path = os.path.join(ad_path, file_name)
        # 读取wav文件数据
        y, sr = librosa.load(file_path, sr=None)
        # 使用窗口批量读取数据
        windows = librosa.util.frame(y, frame_length=window_length, hop_length=hop_length)
        # 处理读取的数据
        print(windows.shape)
        for window in windows.T:
            temp_features=smile.process_signal(window, sr)
            all_features.append(temp_features)
            pass
        print(len(all_features))
        print(all_features[50].shape)

# 将数据类型转换为浮点型
all_features = np.array(all_features)
all_features = all_features.astype(float)
print(all_features.shape)# (****,88)
all_features=np.squeeze(all_features)
#给ad数据加标签
label_ad=np.ones((all_features.shape[0],1))
ad_datasets=np.concatenate((all_features, label_ad), axis=1)





# 定义文件夹路径
cn_path = "./Machine learning/Dataset/diagnosis/train/seg_audio/cn"


# 创建空数组,能够容纳所有特征
all_features = []

# 遍历文件夹中所有wav文件，并读取数据
for file_name in os.listdir(cn_path):
    if file_name.endswith(".wav"):
        file_path = os.path.join(cn_path, file_name)
        # 读取wav文件数据
        y, sr = librosa.load(file_path, sr=None)
        # 使用窗口批量读取数据
        windows = librosa.util.frame(y, frame_length=window_length, hop_length=hop_length)
        # 处理读取的数据
        print(windows.shape)
        for window in windows.T:
            temp_features=smile.process_signal(window, sr)
            all_features.append(temp_features)
            pass
        print(len(all_features))

# 将数据类型转换为浮点型
all_features = np.array(all_features)
all_features = all_features.astype(float)
print(all_features.shape)# (****,88)
all_features=np.squeeze(all_features)

#给cn数据加标签
label_cn=np.zeros((all_features.shape[0],1))
cn_datasets=np.concatenate((all_features, label_cn), axis=1)


# 合并总数据集
all_datasets=np.concatenate((ad_datasets, cn_datasets), axis=0)

np.savetxt('all_seg_datasets.csv', all_datasets, delimiter=',')