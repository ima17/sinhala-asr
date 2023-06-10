from datasets import Dataset, load_dataset
import re

import torchaudio
import librosa
import numpy as np
import torch

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from datasets import load_dataset

# Load the dataset
dataset = load_dataset("keshan/multispeaker-tts-sinhala")

# Define the regular expression pattern to remove special characters
chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�]'

def remove_special_characters(batch):
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower() + " "
    return batch

def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch["audio_path"])
    batch["speech"] = speech_array[0].numpy()
    batch["sampling_rate"] = sampling_rate
    batch["target_text"] = batch["sentence"]
    return batch

def resample(batch):
    batch["speech"] = librosa.resample(np.asarray(batch["speech"]), 48_000, 16_000)
    batch["sampling_rate"] = 16_000
    return batch

def prepare_dataset(batch):
    # Check that all files have the correct sampling rate
    assert (len(set(batch["sampling_rate"])) == 1), f"Make sure all inputs have the same sampling rate."

    batch["input_values"] = batch["speech"]
                
    batch["labels"] = batch["target_text"]
    return batch

# Map the preprocessing functions to the dataset
dataset = dataset.map(remove_special_characters)
dataset = dataset.map(speech_file_to_array_fn, num_proc=64)
dataset = dataset.map(resample, num_proc=64)
dataset = dataset.map(prepare_dataset, batch_size=8, num_proc=4, batched=True, return_tensors="pt")

# Set the format of the dataset to PyTorch tensors
dataset = dataset.with_format("torch")

# Save the processed dataset
torch.save(dataset, 'processed_dataset.pt')
