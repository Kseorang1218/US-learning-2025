# databuilder.py

import os
import pandas as pd
import numpy as np
import librosa
from box import Box
from typing import List, Tuple
import torch

def make_dataframe(config: Box, dirs: List):
    df = {}
    df['path'] = []
    df['data'] = []
    df['machinetype'] = []
    df['modelID'] = []
    df['label'] = []
    df['stft'] = []

    label_map = {
        ("fan", "normal"): 0,
        ("pump", "normal"): 0,
        ("slider", "normal"): 0,
        ("valve", "normal"): 0,
        ("fan", "abnormal"): 1,
        ("pump", "abnormal"): 2,
        ("slider", "abnormal"): 3,
        ("valve", "abnormal"): 4,
    }

    for directory in dirs:
        machinetype = directory.split('/')[-2]
        print(f"making {machinetype} dataframe")
        with os.scandir(directory) as files:
            files = list(files) 
            for file in files:
                filepath = file.path

                data, sample_rate = librosa.load(file, sr=config.sr)
                df["path"].append(filepath)

                D = librosa.stft(data, n_fft=512, hop_length=256, win_length=512)
                magnitude = np.abs(D)
                magnitude = torch.tensor(magnitude, dtype=torch.float32).unsqueeze(0)
                df['stft'].append(magnitude)
                # log_mag = librosa.amplitude_to_db(magnitude, ref=np.max)

                filepath_parts = filepath.split('/')
                model_id = int(filepath_parts[-1].split('_')[2])
                label = filepath_parts[-1].split('_')[0]

                df["data"].append(data)
                df["machinetype"].append(machinetype)
                df["modelID"].append(model_id)
                df["label"].append(label_map[(machinetype, label)])

    dataframe = pd.DataFrame(df)

    return dataframe
    

def data_sampling(data: np.ndarray, sample_size: int, overlap: int, window: int = 3) -> List:
    data_list = []
    for i in range(0, len(data)-sample_size, overlap):
        data_seg = data[i : i + sample_size]
        data_list.append(data_seg)

    return data_list
        
        
def get_data_label_arrays(
        df: pd.DataFrame, sample_size: int, overlap: int
        )-> Tuple[np.ndarray, np.ndarray]:
    data_list = []
    label_list = []

    for _, row in df.iterrows():
        segments = data_sampling(row['data'], sample_size, overlap)
        label = row['label']
        
        for seg in segments:
            data_list.append(seg)        
            label_list.append(label)

    return np.array(data_list), np.array(label_list)