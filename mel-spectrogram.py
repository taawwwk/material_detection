import os
import librosa
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

'''
파형데이터 -> Mel-Spectogram 변환 테스트
'''
plt.rc('font', family='Times New Roman')

steel_data = os.getcwd() + '/sound/steel/steel_2_part25.wav'
wooden_data = os.getcwd() + '/sound/wooden/wooden_2_part25.wav'
glass_data = os.getcwd() + '/sound/glass/glass_2_part25.wav'

test_data = os.getcwd() + '/test/steel/1.0s.wav'
y, sr = librosa.load(test_data, sr=48000)
time = np.arange(len(y))/sr

font_path = '/System/Library/Fonts/Supplemental/Times New Roman.ttf'
fontprop = font_manager.FontProperties(fname=font_path)

plt.figure(figsize=(6, 6))
plt.plot(time, y, linewidth = 1)
plt.xlabel('Time (s)', fontsize=14, fontweight='bold')
plt.ylabel('Amplitude', fontsize=14, fontweight='bold')
plt.xticks(np.arange(0, max(time), 0.2), fontsize=12)
plt.yticks(fontsize=12)
plt.savefig('waveform_plot.pdf', format='pdf', bbox_inches='tight')

print(f'y shape: {y.shape}')
print(f'sr: {sr}')
print(f'y: {y}')

n_mels=128
fixed_length=146

# 오디오 -> 2D 이미지 변환
# Mel-Spectogram: 주파수 변화를 시간에 따라 표현한 그래프 -> 오디오의 특징을 담은 2D Image
# n_mels = 128: 주파수를 128개 구간으로 나눠서 분석
mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)

# power_to_db(): 사람이 듣기 쉽게 로그 스케일(dB)로 변환
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

print(f'mel_spec.shape: {mel_spec.shape}')
print(f'mel_spec.shape: {mel_spec}')
print(f'mel_spec_db.shape: {mel_spec_db.shape}')
print(f'mel_spec_db: {mel_spec_db}')

# Mel-spectrogram plot with custom axis limits and label styles
plt.figure(figsize=(6, 6))
librosa.display.specshow(mel_spec_db, sr=48000, x_axis='time', y_axis='mel')

plt.colorbar(format='%+2.0f dB')
plt.xlabel('Time (s)', fontsize=14, fontweight='bold')
plt.ylabel('Mel Frequency', fontsize=14, fontweight='bold')
plt.xticks(np.arange(0, max(time), 0.2), fontsize=12)
plt.yticks(fontsize=12)
# plt.savefig('mel-spectrogram_plot.pdf', format='pdf', bbox_inches='tight')
plt.close()
# plt.show()