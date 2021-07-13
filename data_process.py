from datasets import Dataset
import pandas as pd
import re

import torchaudio
import librosa
import numpy as np
import torch

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

train_data = Dataset.from_pandas(train_df)
test_data = Dataset.from_pandas(test_df)

train_data = train_data.remove_columns(["full", "x", "filename"])
test_data = test_data.remove_columns(["full", "x", "filename"])

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�]'

def remove_special_characters(batch):
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower() + " "
    return batch

train_data = train_data.map(remove_special_characters)
test_data = test_data.map(remove_special_characters)

def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch["file"])
    batch["speech"] = speech_array[0].numpy()
    batch["sampling_rate"] = sampling_rate
    batch["target_text"] = batch["sentence"]
    return batch

train_data = train_data.map(speech_file_to_array_fn, remove_columns=train_data.column_names, num_proc=64)
test_data = test_data.map(speech_file_to_array_fn, remove_columns=test_data.column_names, num_proc=64)

def resample(batch):
    batch["speech"] = librosa.resample(np.asarray(batch["speech"]), 48_000, 16_000)
    batch["sampling_rate"] = 16_000
    return batch

train_data = train_data.map(resample, num_proc=64)
test_data = test_data.map(resample, num_proc=64)

def prepare_dataset(batch):
    # check that all files have the correct sampling rate
    assert (len(set(batch["sampling_rate"])) == 1), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}."

    batch["input_values"] = processor(batch["speech"], sampling_rate=batch["sampling_rate"][0]).input_values
                    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["target_text"]).input_ids
    return batch

train_data = train_data.map(prepare_dataset, remove_columns=train_data.column_names, batch_size=8, num_proc=4, batched=True, return_tensors="pt")
test_data = test_data.map(prepare_dataset, remove_columns=test_data.column_names, batch_size=8, num_proc=4, batched=True, return_tensors="pt")

train_data = train_data.with_format("torch")
test_data = test_data.with_format("torch")

torch.save(train_data, 'train_si_asr.pt')
torch.save(test_data, 'test_si_asr.pt')
