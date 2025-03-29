import tensorflow as tf
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

# 데이터 로드 및 전처리
def extract_melspectrogram(file_path, n_mels=128, fixed_length=128):
    y, sr = librosa.load(file_path, sr=22050)

    # plt.plot(y)
    # plt.title(f'{file_path}')
    # plt.xlabel('Time')
    # plt.ylabel('Amplitude')
    # plt.show()

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # 크기 맞추기 (Padding or Truncation)
    if mel_spec_db.shape[1] < fixed_length:
        pad_width = fixed_length - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
    else:
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
    plt.savefig(file_name, bbox_inches='tight')
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

file_path = os.path.join(os.getcwd())
wooden_dir = file_path + '/sound/wooden/'
steel_dir = file_path + '/sound/steel/'
glass_dir = file_path + '/sound/glass/'

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
X = np.array(steel_data + wooden_data + glass_data)
y = np.array(steel_labels + wooden_labels + glass_labels)

# 입력 차원 맞추기
X = X[..., np.newaxis]  # CNN 입력을 위한 차원 추가

# 데이터셋 분할 (Train/Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 컴파일 및 학습
model = build_model((X.shape[1], X.shape[2], 1))  # Mel-Spectrogram 입력
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 학습 후 학습 과정 저장
history = model.fit(X_train, y_train, epochs=30, batch_size=8, validation_data=(X_test, y_test))

plot_training_history(history, "training_plot_model_v2.png")