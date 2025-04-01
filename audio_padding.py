import os
import numpy as np
import librosa
import soundfile as sf

# 입력 및 출력 폴더 설정
file_path = os.path.join(os.getcwd())
wooden_dir = file_path + '/padding/wooden/'
steel_dir = file_path + '/padding/steel/'
glass_dir = file_path + '/padding/glass/'

output_folder = file_path + "/padding"

os.makedirs(output_folder, exist_ok=True)

TARGET_DURATION = 3  # 목표 길이 (초)
SAMPLE_RATE = 22050  # 오디오 샘플링 레이트


def process_audio(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    duration = len(y) / sr  # 초 단위 길이
    file_name = os.path.basename(file_path).replace(".wav", "")

    if duration > TARGET_DURATION:
        # 3초씩 분할
        num_splits = int(np.ceil(duration / TARGET_DURATION))
        for i in range(num_splits):
            start_sample = i * TARGET_DURATION * sr
            end_sample = min((i + 1) * TARGET_DURATION * sr, len(y))
            chunk = y[start_sample:end_sample]
            output_file = os.path.join(output_folder, f"{file_name}_part{i + 1}.wav")
            sf.write(output_file, chunk, sr)
            print(f"Saved: {output_file}")

    elif duration < TARGET_DURATION:
        # 기존 사운드를 반복 복사하여 3초로 채우기
        repeat_audio = np.tile(y, int(np.ceil(TARGET_DURATION / duration)))[:TARGET_DURATION * sr]
        output_file = os.path.join(output_folder, f"{file_name}_padded.wav")
        sf.write(output_file, repeat_audio, sr)
        print(f"Saved (padded with repetition): {output_file}")

    else:
        # 그대로 저장
        output_file = os.path.join(output_folder, f"{file_name}.wav")
        sf.write(output_file, y, sr)
        print(f"Saved (original): {output_file}")


# 폴더 내 모든 WAV 파일 처리
for directory in [steel_dir, wooden_dir, glass_dir]:
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            process_audio(os.path.join(directory, filename))

print("Processing complete!")