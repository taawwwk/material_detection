import datetime
import tensorflow as tf
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
from keras.models import load_model
from collections import defaultdict
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Utility Functions

'''*****************************************
설명: wav 파일을 mel-spectrogram으로 전처리함.

input: 
    1. file_path: 파일 경로
    2. n_mels: 주파수 나누는 구간
    3. fixed_length: 데이터 크기 -> CNN의 모든 이미지 크기를 맞추기 위함.

output: 
    1. mel_spec_db: 
*****************************************'''

def extract_melspectrogram(file_path, n_mels, fixed_length, sr=48000):
    # 48kHz로 샘플링 (샘플링: 연속적인 오디오 신호를 일정한 간격으로 데이터(샘플)을 추출하는 과정)
    # y: 오디오 신호 배열 (1D), sr: 초당 샘플링 개수
    y, sr = librosa.load(file_path, sr=48000)

    '''
    # 오디오 -> 2D 이미지 변환
    Mel-Spectogram: 주파수 변화를 시간에 따라 표현한 그래프 -> 오디오의 특징을 담은 2D Image
    n_mels = 128: 주파수를 128개 구간으로 나눠서 분석
    power_to_db(): 로그 스케일(dB)로 변환
    '''
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)


    # fixed_length에 맞춰서 크기 조정하기
    # fixed_length보다 크면 뒷 부분 삭제, 작으면 0으로 뒷부분 채우기
    if mel_spec_db.shape[1] < fixed_length: # 짧은 길이의 오디오는 0을 추가함 (test)
        pad_width = fixed_length - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
    else: # 긴 오디오는 일정 길이만 잘라서 사용 (truncation)
        mel_spec_db = mel_spec_db[:, :fixed_length]

    return mel_spec_db

'''*****************************************
설명: 오디오 데이터가 저장된 폴더에서 파일 가져오기.
     모델 학습에 필요한 데이터, 라벨을 설정.

input: 
    1. data_dir: 디렉토리 경로
    2. label: 물질 이름(ex. wood, steel, glass, ...)

output: 
    1. data: 학습에 필요한 데이터
    2. labels: 학습에 필요한 라벨(클래스)
*****************************************'''
def load_data(data_dir, label, n_mels, fixed_length, sr):
    data = []
    labels = []
    for file_name in sorted(os.listdir(data_dir)):
        file_path = os.path.join(data_dir, file_name)
        if not os.path.isfile(file_path):
            continue
        if file_name.endswith('.wav'): # 파일 확장자가 wav인 파일만 가져오기
            mel_spec = extract_melspectrogram(file_path, n_mels, fixed_length, sr) # mel spectrogram 변환
            data.append(mel_spec)
            labels.append(label)

    print(f'data length: {len(data)}')
    print(f'labels length: {len(labels)}')

    return data, labels

'''*****************************************
설명: waveform에서 mel-spectrogram으로 변환했을 때 생성되는 데이터(이미지)를 저장하여 확인하기 위함.

input: 
    1. mel_spec: mel-spectrogram
    2. file_name: 저장할 파일 이름

output: 
    1. image/mel 경로에 파일로 저장
*****************************************'''
def save_spectrogram_image(mel_spec, file_name):
    plt.figure(figsize=(6, 6))
    librosa.display.specshow(mel_spec, sr=48000, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.savefig(file_path + '/images/mel/' + file_name, bbox_inches='tight')
    plt.close()

'''*****************************************
설명: 2D-CNN 모델 아키텍처 생성

input: 
    1. input_shape: 학습 데이터 크기 지정.

output: 
    1. model: 모델 아키텍처
*****************************************'''
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
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(3, activation='softmax')  # units: 출력값 개수, activation: 활성화 함수
    ])
    model.summary() # 생성한 모델 아키텍처 출력
    return model


'''*****************************************
설명: 모델 학습 과정 그래프 생성 (정확도, 손실값 두 그래프를 만듦)
     x축: Epoch
     y축: Accuracy, Loss

input: 
    1. history: 학습한 모델 객체
    2. test_loss: 학습한 모델의 loss 값
    3. test_accuracy: 학습한 모델의 accuracy 값
    4. file_name: 저장할 그래프 이름

output: 
    1. image/mel 경로에 파일로 저장
*****************************************'''
def plot_training_history(history, test_loss, test_accuracy, file_name):
    fig, ax = plt.subplots(2, 1, figsize=(8, 6))

    # Accuracy 그래프
    ax[0].plot(history.history['accuracy'], label='Train Accuracy')
    ax[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax[0].axhline(y=test_accuracy, color='r', linestyle='--', label='Test Accuracy')
    ax[0].set_title('Model Accuracy')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend()

    # Loss 그래프
    ax[1].plot(history.history['loss'], label='Train Loss')
    ax[1].plot(history.history['val_loss'], label='Validation Loss')
    ax[1].axhline(y=test_loss, color='r', linestyle='--', label='Test Loss')
    ax[1].set_title('Model Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].legend()

    plt.tight_layout()
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()

'''*****************************************
설명: 모델에 대한 혼동행렬 생성 함수

input: 
    1. y_true: 실제 값
    2. y_pred: 모델이 예측한 값
    3. classes: 예측할 클래스
    4. title: 혼동행렬 그림 제목 
    5. filename: 저장할 혼동핼렬 파일 제목
    6. cmap: 혼동행렬 색

output: 
    1. 파일로 저장
*****************************************'''
def plot_confusion_matrix(y_true, y_pred, classes, title, filename, cmap='Blues'):
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=cmap,
                xticklabels=classes,
                yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

'''*****************************************
설명: 2D-CNN 학습 함수

input: 
    1. x_train, y_train: 학습(train), 검증(validate) 입력 데이터
    2. x_val, y_val: 학습(train), 검증(validate) 라벨 데이터
    3. x_test, y_test: 학습 후 테스트(test) 입력, 라벨 데이터
    4. input_shape: 입력 데이터 크기
    5. classes: 라벨(클래스) 종류
    6. file_path: 학습 모델 저장 경로
    7. now_time: 학습 모델 파일명에 학습한 시간을 기입

output: 
    1. 모델 학습 후 저장
*****************************************'''
def train_cnn_model(X_train, y_train, X_val, y_val, X_test, y_test, input_shape, classes, file_path, now_time):
    cnn_model = build_model(input_shape)  # Input Type: Mel Spectrogram(2D)
    cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history = cnn_model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_val, y_val))

    test_loss, test_accuracy = cnn_model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy}, Test Loss: {test_loss}")
    print("**************************************************************************************************************")

    cnn_model.save(file_path+'/model/cnn_'+ now_time +'.h5')

    y_pred = cnn_model.predict(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)

    print("Classification Report:")
    print(classification_report(y_test, y_pred_labels, target_names=classes))
    print("**************************************************************************************************************")

    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    for true, pred in zip(y_test, y_pred_labels):
        class_total[true] += 1
        if true == pred:
            class_correct[true] += 1

    per_class_accuracy = {i: (class_correct[i] / class_total[i]) * 100 if class_total[i] > 0 else 0 for i in range(len(classes))}

    plot_confusion_matrix(y_test, y_pred_labels, classes, 'CNN Confusion Matrix', "graph/confusion_matrix_" + 'cnn_' + now_time + ".png", cmap='Blues')

    plot_training_history(history, test_loss, test_accuracy, "graph/training_plot_" + now_time + ".png")

    results_dir = os.path.join(file_path, 'results')
    os.makedirs(results_dir, exist_ok=True)
    np.save(os.path.join(results_dir, f'y_test_cnn_{now_time}.npy'), y_test)
    np.save(os.path.join(results_dir, f'y_pred_cnn_{now_time}.npy'), y_pred_labels)
    print(f"Saved CNN test labels and predictions to {results_dir}")
    print("**************************************************************************************************************")

'''*****************************************
설명: SVM 학습 함수

input: 
    1. x_train_flat, y_train_flat: 학습(train), 검증(validate) 입력 데이터
    2. x_val_flat, y_val_flat: 학습(train), 검증(validate) 라벨 데이터
    3. x_test_flat, y_test_flat: 학습 후 테스트(test) 입력, 라벨 데이터
    4. input_shape: 입력 데이터 크기
    5. classes: 라벨(클래스) 종류
    6. file_path: 학습 모델 저장 경로
    7. now_time: 학습 모델 파일명에 학습한 시간을 기입

output: 
    1. 모델 학습 후 저장
*****************************************'''
def train_svm_model(X_train_flat, y_train_flat, X_val_flat, y_val_flat, X_test_flat, y_test_flat, classes, file_path, now_time):
    scaler = StandardScaler()
    X_train_flat = scaler.fit_transform(X_train_flat)
    X_val_flat   = scaler.transform(X_val_flat)
    X_test_flat  = scaler.transform(X_test_flat)
    joblib.dump(scaler, file_path + '/model/scaler_' + now_time + '.joblib')

    svm_model = SVC(kernel='rbf', C=1.0)
    svm_model.fit(X_train_flat, y_train_flat)
    joblib.dump(svm_model, file_path + '/model/svm_' + now_time + '.joblib')

    svm_train_acc = svm_model.score(X_train_flat, y_train_flat)
    svm_test_acc = svm_model.score(X_test_flat, y_test_flat)
    print(f"SVM Training Accuracy: {svm_train_acc:.4f}")
    print("**************************************************************************************************************")
    print(f"SVM Test Accuracy: {svm_test_acc:.4f}")
    print("**************************************************************************************************************")

    y_pred_svm = svm_model.predict(X_test_flat)

    plot_confusion_matrix(y_test_flat, y_pred_svm, classes, 'SVM Confusion Matrix', "graph/confusion_matrix_svm_" + now_time + ".png", cmap='Purples')

    results_dir = os.path.join(file_path, 'results')
    os.makedirs(results_dir, exist_ok=True)
    np.save(os.path.join(results_dir, f'y_test_svm_{now_time}.npy'), y_test_flat)
    np.save(os.path.join(results_dir, f'y_pred_svm_{now_time}.npy'), y_pred_svm)
    print(f"Saved SVM test labels and predictions to {results_dir}")
    print("**************************************************************************************************************")

'''*****************************************
설명: MLP 학습 함수

input: 
    1. x_train_flat, y_train_flat: 학습(train), 검증(validate) 입력 데이터
    2. x_val_flat, y_val_flat: 학습(train), 검증(validate) 라벨 데이터
    3. x_test_flat, y_test_flat: 학습 후 테스트(test) 입력, 라벨 데이터
    4. input_shape: 입력 데이터 크기
    5. classes: 라벨(클래스) 종류
    6. file_path: 학습 모델 저장 경로
    7. now_time: 학습 모델 파일명에 학습한 시간을 기입

output: 
    1. 모델 학습 후 저장
*****************************************'''
def train_mlp_model(X_train_flat, y_train_flat, X_val_flat, y_val_flat, X_test_flat, y_test_flat, classes, file_path, now_time):
    scaler = StandardScaler()
    X_train_flat = scaler.fit_transform(X_train_flat)
    X_val_flat   = scaler.transform(X_val_flat)
    X_test_flat  = scaler.transform(X_test_flat)
    joblib.dump(scaler, file_path + '/model/keras_scaler_' + now_time + '.joblib')

    class_names = list(classes)
    keras_mlp_model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train_flat.shape[1],)),
        Dense(64, activation='relu'),
        # Dropout(0.3),
        Dense(len(class_names), activation='softmax')
    ])
    keras_mlp_model.compile(optimizer='adam',
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])

    y_train_onehot = tf.keras.utils.to_categorical(y_train_flat, num_classes=len(class_names))
    y_val_onehot = tf.keras.utils.to_categorical(y_val_flat, num_classes=len(class_names))
    y_test_onehot = tf.keras.utils.to_categorical(y_test_flat, num_classes=len(class_names))

    history_keras_mlp = keras_mlp_model.fit(
        X_train_flat, y_train_onehot,
        validation_data=(X_val_flat, y_val_onehot),
        epochs=20,
        batch_size=16
    )

    test_loss_keras_mlp, test_acc_keras_mlp = keras_mlp_model.evaluate(X_test_flat, y_test_onehot)
    print(f"Keras MLP Test Accuracy: {test_acc_keras_mlp:.4f}, Test Loss: {test_loss_keras_mlp:.4f}")
    print("**************************************************************************************************************")

    keras_mlp_model.save(file_path + '/model/keras_mlp_' + now_time + '.h5')

    y_pred_keras_mlp = keras_mlp_model.predict(X_test_flat)
    y_pred_labels_keras_mlp = np.argmax(y_pred_keras_mlp, axis=1)
    y_true_labels = np.argmax(y_test_onehot, axis=1)

    print("Keras MLP Classification Report:")
    print(classification_report(y_true_labels, y_pred_labels_keras_mlp, target_names=class_names))
    print("**************************************************************************************************************")

    plot_confusion_matrix(y_true_labels, y_pred_labels_keras_mlp, class_names, 'Keras MLP Confusion Matrix', "graph/confusion_matrix_keras_mlp_" + now_time + ".png", cmap='Greens')

    results_dir = os.path.join(file_path, 'results')
    os.makedirs(results_dir, exist_ok=True)
    np.save(os.path.join(results_dir, f'y_test_keras_mlp_{now_time}.npy'), y_true_labels)
    np.save(os.path.join(results_dir, f'y_pred_keras_mlp_{now_time}.npy'), y_pred_labels_keras_mlp)
    print(f"Saved Keras MLP test labels and predictions to {results_dir}")
    print("**************************************************************************************************************")

'''*****************************************
메인함수

output:
    1. CNN, SVM, MLP 학습 모델 -> model 디렉토리에 저장
    2. 라벨 인코더 -> model 디렉토리에 저장
    3. 학습 그래프, 혼동행렬 -> graph 디렉토리에 저장
*****************************************'''
now_time = str(int(datetime.datetime.now().timestamp()))

# 프로젝트 위치 지정
file_path = os.path.join(os.getcwd())

# 재질 사운드 데이터 경로
wooden_dir = file_path + '/sound/wooden/'
steel_dir = file_path + '/sound/steel/'
glass_dir = file_path + '/sound/glass/'

# 라벨 리스트
class_names = ['steel', 'wooden', 'glass']

# 데이터 로드
steel_data, steel_labels = load_data(steel_dir, 'steel',  n_mels=128, fixed_length=146, sr = 48000)
wooden_data, wooden_labels = load_data(wooden_dir, 'wooden',  n_mels=128, fixed_length=146, sr = 48000)
glass_data, glass_labels = load_data(glass_dir, 'glass',  n_mels=128, fixed_length=146, sr = 48000)

# 문자열 라벨 -> 정수 인코딩
all_labels = steel_labels + wooden_labels + glass_labels
le = LabelEncoder()
y = le.fit_transform(all_labels)
joblib.dump(le, file_path + '/model/label_encoder_' + now_time + '.joblib')

# 데이터 변환
X = np.array(steel_data + wooden_data + glass_data) # (샘플개수, 128, 128)
X = X[..., np.newaxis]  # CNN 입력을 위한 채널 차원 추가 (샘플개수, 128, 128, 1)


# 데이터셋 분할 (Train/Test)
# 학습:테스트:검증 = 5:3:2
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.5, random_state=42)  # 50% train, 50% temp
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.6, random_state=42)  # 60% test, 40% validation
print("학습 데이터 크기: " + str(len(X_train)) + " 검증 데이터 크기: " + str(len(X_val)) + " 테스트 데이터 크기: " + str(len(X_test)))
print("**************************************************************************************************************")

# SVM, MLP 학습을 위해 데이터를 평탄화
X_flat = X.reshape((X.shape[0], -1))

# Split data for flat arrays
# 데이터셋 분할 (Train/Test)
# 학습:테스트:검증 = 5:3:2
X_train_flat, X_temp_flat, y_train_flat, y_temp_flat = train_test_split(
    X_flat, y, test_size=0.5, random_state=42)
X_val_flat, X_test_flat, y_val_flat, y_test_flat = train_test_split(
    X_temp_flat, y_temp_flat, test_size=0.6, random_state=42)

# CNN 학습
train_cnn_model(X_train, y_train, X_val, y_val, X_test, y_test, (X.shape[1], X.shape[2], 1), class_names, file_path, now_time)

# SVM 학습
train_svm_model(X_train_flat, y_train_flat, X_val_flat, y_val_flat, X_test_flat, y_test_flat, class_names, file_path, now_time)

# MLP 학습
train_mlp_model(X_train_flat, y_train_flat, X_val_flat, y_val_flat, X_test_flat, y_test_flat, le.classes_, file_path, now_time)
