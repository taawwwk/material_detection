import os
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

# === 설정 ===
TEST_AUDIO_DIR = './sound'
SEGMENT_DIR = 'pattern_sounds'
SAMPLE_RATE = 48000
SEGMENT_DURATIONS = [0.5 * i for i in range(1, 11)]  # 0.5초 ~ 5.0초
N_MELS = 128
HOP_LENGTH = 512
EXPECTED_WIDTH = 146  # CNN input의 시간축 크기, 3초 기준값이었으면 이걸 유지 or 동적 처리 가능

# === 1단계: 슬라이싱 함수 ===
def slice_and_save_segments(filepath, output_base_dir, segment_durations, sample_rate):
    filename = os.path.basename(filepath)
    label = os.path.splitext(filename)[0]  # 예: glass, steel, wooden
    y, sr = librosa.load(filepath, sr=sample_rate)

    os.makedirs(os.path.join(output_base_dir, label), exist_ok=True)

    for dur in segment_durations:
        num_samples = int(dur * sample_rate)
        y_trimmed = y[:num_samples]
        output_path = os.path.join(output_base_dir, label, f'{dur:.1f}s.wav')
        sf.write(output_path, y_trimmed, sample_rate)
        print(f'Saved: {output_path}')

# === 2단계: Mel-Spectrogram 변환 ===
def audio_to_melspectrogram(filepath, duration_sec=2.0):
    max_length_samples = int(duration_sec * SAMPLE_RATE)
    y, sr = librosa.load(filepath, sr=SAMPLE_RATE)

    if len(y) < max_length_samples:
        y = np.pad(y, (0, max_length_samples - len(y)), mode='constant')
    else:
        y = y[:max_length_samples]

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db

# 정규화
def normalize_spectrogram(spec):
    return (spec - np.min(spec)) / (np.max(spec) - np.min(spec))

# 크기 맞추기 (Padding or Truncating)
def resize_spec(spec, expected_width):
    if spec.shape[1] > expected_width:
        return spec[:, :expected_width]
    elif spec.shape[1] < expected_width:
        pad_width = expected_width - spec.shape[1]
        return np.pad(spec, ((0, 0), (0, pad_width)), mode='constant')
    return spec

# === 3단계: 전체 Mel 처리 ===
def process_all_segments(segment_dir):
    X_data = []
    file_info = []  # (label, duration, filepath)

    for label in os.listdir(segment_dir):
        label_path = os.path.join(segment_dir, label)
        if not os.path.isdir(label_path):
            continue
        for fname in sorted(os.listdir(label_path)):
            if fname.endswith('.wav'):
                fpath = os.path.join(label_path, fname)
                mel = audio_to_melspectrogram(fpath, duration_sec=2.0)
                mel = normalize_spectrogram(mel)
                mel = resize_spec(mel, expected_width=EXPECTED_WIDTH)
                X_data.append(mel[..., np.newaxis])  # CNN 입력 형태 (128, ?, 1)
                duration = fname.replace('s.wav', '')
                file_info.append((label, duration, fpath))

    return np.array(X_data), file_info

# === 실행 ===
if __name__ == '__main__':
    # Step 1: 슬라이싱
    for fname in os.listdir(TEST_AUDIO_DIR):
        if fname.endswith('.wav'):
            full_path = os.path.join(TEST_AUDIO_DIR, fname)
            slice_and_save_segments(full_path, SEGMENT_DIR, SEGMENT_DURATIONS, SAMPLE_RATE)

    # Step 2: Mel-Spectrogram 변환
    X_test, info = process_all_segments(SEGMENT_DIR)
    print(f'\n✅ 변환 완료: 총 {len(X_test)}개 Mel-Spectrogram 생성됨.')

    # (선택) 시각화 예시
    # import matplotlib.pyplot as plt
    # plt.imshow(X_test[0].squeeze(), origin='lower', aspect='auto', cmap='magma')
    # plt.title(f"{info[0]}")
    # plt.colorbar()
    # plt.show()