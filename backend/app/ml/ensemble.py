# ensemble.py
import joblib
import torch
import numpy as np
from .train_rnn import SimpleRNN

def predict_rnn(seq, model_path='rnn_model.pth'):
    model = SimpleRNN(n_features=seq.shape[1])
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    with torch.no_grad():
        x = torch.tensor(seq[None].astype('float32'))
        return model(x).item()

def predict_svr(features, model_path='svr_model.joblib'):
    model = joblib.load(model_path)
    return float(model.predict([features])[0])

def ensemble_predict(seq, flat_features):
    try:
        rnn_pred = predict_rnn(np.asarray(seq), model_path='rnn_model.pth')
    except Exception:
        rnn_pred = 0.0
    try:
        svr_pred = predict_svr(flat_features, model_path='svr_model.joblib')
    except Exception:
        svr_pred = 0.0
    return (rnn_pred + svr_pred) / (2 if (rnn_pred and svr_pred) else 1)
