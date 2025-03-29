import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# 매질별 데이터 경로
materials = ["steel", "wooden", "glass"]
base_dir = os.path.join(os.getcwd()) + "/sound"
wooden_dir = base_dir + '/wooden/'
steel_dir = base_dir + '/steel/'
glass_dir = base_dir + '/glass/'

# 전체 데이터 로드 함수
def load_full_dataset():
    data, labels = [], []
    for material in materials:
        X, y = load_dataset(material)
        data.append(X)
        labels.append(y)
    return np.vstack(data), np.hstack(labels)

# 특징 추출 함수 (MFCC + Mel-Spectrogram)
def extract_features(file_path, n_mfcc=13, n_mels=128):
    y, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return np.mean(mfcc, axis=1), np.mean(mel_spec_db, axis=1)

# 데이터 로드 함수
def load_dataset(material):
    data, labels = [], []
    folder_path = os.path.join(base_dir, material)
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.wav'):
            file_path = os.path.join(folder_path, file_name)
            mfcc, mel_spec = extract_features(file_path)
            feature_vector = np.concatenate((mfcc, mel_spec))
            data.append(feature_vector)
            labels.append(materials.index(material))
    return np.array(data), np.array(labels)

# 모델 학습 함수 (SVM)
def train_svm(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = SVC(kernel='rbf', C=1.0, gamma='scale')
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"SVM Accuracy: {accuracy:.4f}")
    return model

# 모델 학습 함수 (MLP)
def train_mlp(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', max_iter=500)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"MLP Accuracy: {accuracy:.4f}")
    return model

# 모델 학습 함수 (CNN)
def train_cnn(X_train, y_train, X_test, y_test):
    X_train = X_train.reshape(-1, 16, 16, 1)
    X_test = X_test.reshape(-1, 16, 16, 1)
    y_train = to_categorical(y_train, num_classes=3)
    y_test = to_categorical(y_test, num_classes=3)

    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(16, 16, 1)),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

    # 학습 과정 시각화
    plot_training_history(history, f"{material}_cnn_training.png")

    return model

# 학습 과정 시각화 함수
def plot_training_history(history, file_name):
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()

# 특징 분석 시각화 함수
def plot_feature_distribution(X, y, file_name):
    plt.figure(figsize=(8, 6))
    for i, material in enumerate(materials):
        plt.scatter(X[y == i, 0], X[y == i, 1], label=material)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()

# 모든 모델 학습 실행
def train_all_models():
    X, y = load_full_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for material in materials:
        print(f"Training models for {material}...")

        # 특정 매질 데이터만 필터링
        material_label = materials.index(material)
        X_train_filtered = X_train[y_train == material_label]
        y_train_filtered = y_train[y_train == material_label]
        X_test_filtered = X_test[y_test == material_label]
        y_test_filtered = y_test[y_test == material_label]

        # 최소 2개 이상의 클래스가 있도록 라벨 수정
        y_train_filtered = np.where(y_train_filtered == material_label, 1, 0)
        y_test_filtered = np.where(y_test_filtered == material_label, 1, 0)

        if len(np.unique(y_train_filtered)) < 2:
            print(f"Skipping {material} training: Not enough class variation")
            continue

        # SVM 학습
        # train_svm(X_train_filtered, y_train_filtered, X_test_filtered, y_test_filtered)

        # MLP 학습
        train_mlp(X_train_filtered, y_train_filtered, X_test_filtered, y_test_filtered)

        # CNN 학습
        train_cnn(X_train_filtered, y_train_filtered, X_test_filtered, y_test_filtered)

        # 특징 분석 시각화
        plot_feature_distribution(X_train_filtered, y_train_filtered, f"{material}_features.png")

# 실행
if __name__ == "__main__":
    train_all_models()
