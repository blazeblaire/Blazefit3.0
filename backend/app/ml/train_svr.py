# train_svr.py
import joblib
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def train(path):
    df = pd.read_parquet(path)
    agg = df.groupby('athlete_id').last().reset_index()
    X = agg[[c for c in ['pace','vo2max','height','weight'] if c in agg.columns]].fillna(0)
    y = agg['time']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
    model = SVR(kernel='rbf')
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print('RMSE', np.sqrt(mean_squared_error(y_test, preds)))
    joblib.dump(model, 'svr_model.joblib')

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    args = p.parse_args()
    train(args.data)
