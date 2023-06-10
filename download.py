from sklearn.model_selection import train_test_split

import pandas as pd

cols = ['filename', 'x', 'sentence']
df = pd.read_csv('train.tsv', sep='\t', names=cols, header=None)

def make_path(full):
    return f'asr_sinhala/data/{full[:2]}/{full}.flac'    

df['file'] = df['filename'].apply(make_path)
df = df.loc[~df.sentence.str.contains('\t'), :]

train, test = train_test_split(df, test_size=0.15)

train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)
