import datetime
import tensorflow as tf
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from keras.models import load_model

# 데이터 로드 및 전처리
'''
n_mels: 주파수를 어느 구간만큼 나눌 것인지
fixed_length: 데이터 크기 -> CNN 학습을 위한 모든 이미지의 크기를 맞추기 위함
'''
def extract_melspectrogram(file_path, n_mels=128, fixed_length=128):
    # 22.05kHz로 샘플링
    # 샘플링: 연속적인 오디오 신호를 일정한 간격으로 데이터(샘플)을 추출하는 과정
    y, sr = librosa.load(file_path, sr=22050) # y: 오디오 신호 배열 1D), sr: 초당 샘플링 개수
    '''
    plt.plot(y)
    plt.title(f'{file_path}')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()
    '''

    # 오디오 -> 2D 이미지 변환
    # Mel-Spectogram: 주파수 변화를 시간에 따라 표현한 그래프 -> 오디오의 특징을 담은 2D Image
    # n_mels = 128: 주파수를 128개 구간으로 나눠서 분석
    # power_to_db(): 사람이 듣기 쉽게 로그 스케일(dB)로 변환
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # 크기 맞추기 (Padding or Truncation)
    if mel_spec_db.shape[1] < fixed_length: # 짧은 길이의 오디오는 0을 추가함 (padding)
        pad_width = fixed_length - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
    else: # 긴 오디오는 일정 길이만 잘라서 사용 (truncation)
        mel_spec_db = mel_spec_db[:, :fixed_length]

    return mel_spec_db

def load_data(data_dir, label):
    data = []
    labels = []
    for file_name in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file_name)
        if file_name.endswith('.wav'):
            mel_spec = extract_melspectrogram(file_path)
            data.append(mel_spec)
            labels.append(label)
    print(f'data length: {len(data)}')
    print(f'labels length: {len(labels)}')
    return data, labels

# Mel-Spectrogram 이미지 저장 함수
def save_spectrogram_image(mel_spec, file_name):
    plt.figure(figsize=(6, 4))
    librosa.display.specshow(mel_spec, sr=22050, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Mel-Spectrogram")
    plt.savefig(file_path + '/images/' + file_name, bbox_inches='tight')
    plt.close()

# CNN 모델 구조 정의
def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')  # steel, wooden, glass
    ])
    model.summary()
    return model

# 학습 과정 시각화 및 저장
def plot_training_history(history, file_name):
    fig, ax = plt.subplots(2, 1, figsize=(8, 6))

    # Accuracy 그래프
    ax[0].plot(history.history['accuracy'], label='Train Accuracy')
    ax[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax[0].set_title('Model Accuracy')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend()

    # Loss 그래프
    ax[1].plot(history.history['loss'], label='Train Loss')
    ax[1].plot(history.history['val_loss'], label='Validation Loss')
    ax[1].set_title('Model Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].legend()

    plt.tight_layout()
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()


now_time = str(int(datetime.datetime.now().timestamp()))

file_path = os.path.join(os.getcwd())

wooden_dir = file_path + '/sound_v2/wooden/'
steel_dir = file_path + '/sound_v2/steel/'
glass_dir = file_path + '/sound_v2/glass/'

# 데이터 로드
glass_data, glass_labels = load_data(glass_dir, 2) # 유리 = 2
wooden_data, wooden_labels = load_data(wooden_dir, 1)  # 나무 = 1
steel_data, steel_labels = load_data(steel_dir, 0)  # 철 = 0

# GPU 가속
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

# Mel-Spectrogram 저장 실행 (각 파일별 저장)
for idx, (mel_spec, label) in enumerate(zip(wooden_data, wooden_labels)):
    save_spectrogram_image(mel_spec, f"wooden_{idx+1}.png")

for idx, (mel_spec, label) in enumerate(zip(steel_data, steel_labels)):
    save_spectrogram_image(mel_spec, f"steel_{idx+1}.png")

for idx, (mel_spec, label) in enumerate(zip(glass_data, glass_labels)):
    save_spectrogram_image(mel_spec, f"glass_{idx+1}.png")

# 데이터 변환
X = np.array(steel_data + wooden_data + glass_data) # (샘플개수, 128, 128)
y = np.array(steel_labels + wooden_labels + glass_labels)

# 입력 차원 맞추기
X = X[..., np.newaxis]  # CNN 입력을 위한 채널 차원 추가 (샘플개수, 128, 128, 1)

# 데이터셋 분할 (Train/Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("학습 데이터 크기: " + str(len(X_train)) + "검증 데이터 크기: " + str(len(X_test)))

# 모델 컴파일 및 학습
model = build_model((X.shape[1], X.shape[2], 1))  # Mel-Spectrogram 입력
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 학습 후 학습 과정 저장
history = model.fit(X_train, y_train, epochs=20, batch_size=10, validation_data=(X_test, y_test))

model.save(file_path+'/model/'+now_time+'.h5')

plot_training_history(history, "graph/training_plot_" + now_time + ".png")