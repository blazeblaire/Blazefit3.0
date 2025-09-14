# preprocess.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_and_clean(path):
    df = pd.read_csv(path, parse_dates=['session_date'])
    for col in ['time','distance','wind']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    for col in ['height','weight','vo2max']:
        if col in df.columns:
            df[col] = df.groupby('athlete_id')[col].transform(lambda x: x.fillna(x.median()))
            df[col] = df[col].fillna(df[col].median())

    df['pace'] = df['time'] / df['distance']

    if 'split1' in df.columns and 'split2' in df.columns:
        df['accel'] = (df['split2'] - df['split1']) / df['distance']

    df['injury_flag'] = df.get('injury_history', pd.Series()).fillna('none').apply(lambda x: 1 if x != 'none' else 0) if 'injury_history' in df.columns else 0

    num_cols = [c for c in ['time','distance','wind','height','weight','vo2max','pace'] if c in df.columns]
    scaler = StandardScaler()
    if num_cols:
        df[num_cols] = scaler.fit_transform(df[num_cols])

    return df, scaler

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    df, scaler = load_and_clean(args.input)
    df.to_parquet(args.out)
