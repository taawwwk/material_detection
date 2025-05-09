import os
import numpy as np
import tensorflow as tf
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from scipy.ndimage import zoom
from collections import defaultdict
import joblib
import time
from sklearn.preprocessing import LabelEncoder

# === 설정 ===
CNN_MODEL_PATH = './model/1745329693.h5'
MLP_MODEL_PATH = './model/mlp_1745329693.joblib'
SVM_MODEL_PATH = './model/svm_1745329693.joblib'
SEGMENT_DIR = './test'
SAMPLE_RATE = 48000
N_MELS = 128
HOP_LENGTH = 512
EXPECTED_WIDTH = 146

label_encoder = joblib.load('./model/label_encoder_1745329693.joblib')
class_names = list(label_encoder.classes_)

# === Mel-Spectrogram 전처리 함수 ===
def audio_to_melspectrogram(filepath):
    y, sr = librosa.load(filepath, sr=SAMPLE_RATE)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db

def normalize_spectrogram(spec):
    return (spec - np.min(spec)) / (np.max(spec) - np.min(spec))

def resize_spec_by_scaling(spec, target_width=128):
    current_height, current_width = spec.shape
    zoom_factor = target_width / current_width
    spec_resized = zoom(spec, (1, zoom_factor))  # only scale time-axis
    return spec_resized

def resize_spec(spec, expected_width):
    if spec.shape[1] > expected_width:
        return spec[:, :expected_width]
    elif spec.shape[1] < expected_width:
        pad_width = expected_width - spec.shape[1]
        return np.pad(spec, ((0, 0), (0, pad_width)), mode='constant')
    return spec

def process_all_segments(segment_dir):
    X_data = []
    y_true = []
    file_info = []

    for label in os.listdir(segment_dir):
        label_path = os.path.join(segment_dir, label)
        if not os.path.isdir(label_path):
            continue
        for fname in sorted(os.listdir(label_path)):
            if fname.endswith('.wav'):
                fpath = os.path.join(label_path, fname)
                mel = audio_to_melspectrogram(fpath)
                mel = normalize_spectrogram(mel)
                mel = resize_spec(mel, EXPECTED_WIDTH)
                if label in class_names:
                    y_true.append(label_encoder.transform([label])[0])
                else:
                    print(f"[WARNING] Unknown label '{label}' not in class_names. Skipping file: {fname}")
                    continue
                print(f"[DEBUG] {fname}: mel shape = {mel.shape}, label = {label}")
                X_data.append(mel[..., np.newaxis])
                file_info.append((label, fname, fpath))

    return np.array(X_data), np.array(y_true), file_info

def load_data(data_dir, label):
    data = []
    labels = []

    for file_name in sorted(os.listdir(data_dir)):
        file_path = os.path.join(data_dir, file_name)
        if not os.path.isfile(file_path):
            continue
        if file_name.endswith('.wav'):
            mel = audio_to_melspectrogram(file_path)
            mel = normalize_spectrogram(mel)
            mel = resize_spec(mel, EXPECTED_WIDTH)
            data.append(mel[..., np.newaxis])
            labels.append(label)

    return np.array(data), labels

def save_spectrogram_image(mel_spec, file_name):
    plt.figure(figsize=(6, 6))
    librosa.display.specshow(mel_spec, sr=48000, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    # plt.title("Mel-Spectrogram")
    plt.savefig('./test/mel/' + file_name, bbox_inches='tight')
    plt.close()

# === 예측 및 시각화 ===
def predict_and_visualize():
    print("✅ 모델 불러오는 중...")
    model = tf.keras.models.load_model(CNN_MODEL_PATH)

    print("✅ 데이터 전처리 중...")
    X_test, y_true, file_info = process_all_segments(SEGMENT_DIR)

    print("✅ 예측 수행 중...")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\n🎯 예측 결과:")
    for i, (true_label, fname, _) in enumerate(file_info):
        probs = y_pred_probs[i]
        print(f"{fname} (실제: {true_label}) → softmax = {np.round(probs, 3)}")

    # 정확도 출력
    acc = accuracy_score(y_true, y_pred)
    print(f"\n✅ 전체 정확도: {acc*100:.2f}%")

    # 라벨별 정확도 계산
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    for true, pred in zip(y_true, y_pred):
        class_total[true] += 1
        if true == pred:
            class_correct[true] += 1

    per_class_accuracy = {class_names[i]: (class_correct[i] / class_total[i]) * 100 if class_total[i] > 0 else 0 for i in range(len(class_names))}

    # 막대 그래프 시각화
    # keys = list(per_class_accuracy.keys())
    # values = [per_class_accuracy[k] for k in keys]
    # plt.bar(keys, values, color='skyblue', align='center')
    # plt.xticks(ticks=np.arange(len(keys)), labels=keys)
    # plt.xlabel('Material Label')
    # plt.ylabel('Accuracy (%)')
    # plt.ylim(0, 100)
    # plt.title('Per-Class Accuracy')
    # plt.tight_layout()
    # plt.savefig("graph/per_class_accuracy.png")
    # plt.close()

    # 시간별, 클래스별 정확도 시각화
    label_map = {i: name for i, name in enumerate(class_names)}
    time_accuracy_per_class = defaultdict(lambda: defaultdict(list))

    for (true_label, fname, _), pred_idx in zip(file_info, y_pred):
        time_key = fname.split('.')[0]  # e.g., "0.5s"
        time_accuracy_per_class[time_key][label_encoder.transform([true_label])[0]].append(int(label_encoder.transform([true_label])[0] == pred_idx))

    sorted_keys = sorted(time_accuracy_per_class.keys(), key=lambda x: float(x.replace('s','')))
    x_ticks = [float(k.replace('s','')) for k in sorted_keys]

    plt.figure(figsize=(8, 5))
    for label_idx, label_name in label_map.items():
        y_values = [
            np.mean(time_accuracy_per_class[k][label_idx]) * 100 if label_idx in time_accuracy_per_class[k] else 0
            for k in sorted_keys
        ]
        plt.plot(x_ticks, y_values, marker='o', label=label_name)

    plt.xlabel('Audio Duration (sec)')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy by Audio Duration per Class')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("graph/accuracy_by_duration_per_class.png")
    plt.close()


# === 입력 시간별 모델 평가 함수 ===
def evaluate_models_over_durations():
    print("✅ 모델 불러오는 중...")
    cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
    mlp_model = joblib.load(MLP_MODEL_PATH)
    svm_model = joblib.load(SVM_MODEL_PATH)

    print("✅ 입력 시간별 데이터 전처리 중...")
    X_test, y_true, file_info = process_all_segments(SEGMENT_DIR)

    # duration 정렬을 위한 키 추출
    time_map = defaultdict(list)
    for i, (_, fname, _) in enumerate(file_info):
        dur_key = fname.replace('.wav', '').replace('s', '')  # '0.5' as string
        time_map[dur_key].append(i)

    durations = sorted(time_map.keys(), key=lambda x: float(x))

    for dur in durations:
        idxs = time_map[dur]
        X_dur = X_test[idxs]
        y_dur = y_true[idxs]
        X_flat = X_dur.reshape((X_dur.shape[0], -1))

        print(f"\n⏱ Duration = {dur} ({len(idxs)} samples)")

        # CNN
        start = time.time()
        cnn_pred = np.argmax(cnn_model.predict(X_dur), axis=1)
        cnn_time = (time.time() - start) * 1000 / len(X_dur)
        cnn_acc = accuracy_score(y_dur, cnn_pred)
        print(f"🧠 CNN   | Acc: {cnn_acc:.3f} | Time: {cnn_time:.2f} ms")
        # CNN prediction debug
        print(f"[DEBUG] CNN prediction: {cnn_pred}")
        print(f"[DEBUG] True labels: {y_dur}")

        # MLP
        start = time.time()
        mlp_pred = mlp_model.predict(X_flat)
        mlp_time = (time.time() - start) * 1000 / len(X_dur)
        mlp_acc = accuracy_score(y_dur, mlp_pred)
        print(f"🔧 MLP   | Acc: {mlp_acc:.3f} | Time: {mlp_time:.2f} ms")
        # MLP prediction debug
        print(f"[DEBUG] MLP prediction: {mlp_pred}")

        # SVM
        start = time.time()
        svm_pred = svm_model.predict(X_flat)
        svm_time = (time.time() - start) * 1000 / len(X_dur)
        svm_acc = accuracy_score(y_dur, svm_pred)
        print(f"🔍 SVM   | Acc: {svm_acc:.3f} | Time: {svm_time:.2f} ms")
        # SVM prediction debug
        print(f"[DEBUG] SVM prediction: {svm_pred}")


if __name__ == '__main__':
    evaluate_models_over_durations()