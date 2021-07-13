from sklearn.model_selection import train_test_split

import wget
from tqdm.auto import tqdm
import pandas as pd
import zipfile
import os

#for i in tqdm([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'a', 'b', 'c', 'd', 'e', 'f']):
#    wget.download(f'https://www.openslr.org/resources/52/asr_sinhala_{i}.zip')
#    with zipfile.ZipFile(f"./asr_sinhala_{i}.zip", 'r') as zip_ref:
#        zip_ref.extractall('./')
#    os.remove(f"./asr_sinhala_{i}.zip")

cols = ['filename', 'x', 'sentence']
df = pd.read_csv('../large-sinhala-asr-dataset-1/asr_sinhala/utt_spk_text.tsv', sep='\t', names=cols, header=None)

def make_path(full):
    return f'asr_sinhala/data/{full[:2]}/{full}.flac'    

df['file'] = df['filename'].apply(make_path)
df = df.loc[~df.sentence.str.contains('\t'), :]

train, test = train_test_split(df, test_size=0.15)

train.to_csv('train.tsv', index=False, sep='\t')
test.to_csv('test.tsv', index=False, sep='\t')

