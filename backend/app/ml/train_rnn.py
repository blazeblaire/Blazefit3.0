# train_rnn.py (PyTorch simple RNN trainer)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class TimeSeriesDataset(Dataset):
    def __init__(self, df, seq_len=10, features=None, target='time'):
        self.seq_len = seq_len
        self.features = features or [c for c in df.columns if c not in ['athlete_id','session_date',target]]
        self.target = target
        self.groups = []
        for aid, g in df.groupby('athlete_id'):
            arr = g.sort_values('session_date')
            self.groups.append(arr)

    def __len__(self):
        return sum(max(0, len(g)-self.seq_len) for g in self.groups)

    def __getitem__(self, idx):
        acc = 0
        for g in self.groups:
            L = len(g)
            step = max(0, L-self.seq_len)
            if idx < acc + step:
                start = idx - acc
                seq = g[self.features].iloc[start:start+self.seq_len].values.astype('float32')
                y = g[self.target].iloc[start+self.seq_len]
                return torch.tensor(seq), torch.tensor(y).float()
            acc += step
        raise IndexError

class SimpleRNN(nn.Module):
    def __init__(self, n_features, hidden=64, n_layers=2):
        super().__init__()
        self.rnn = nn.GRU(n_features, hidden, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        return self.fc(out)

def train(df_path, epochs=5):
    df = pd.read_parquet(df_path)
    ds = TimeSeriesDataset(df, seq_len=1)
    dl = DataLoader(ds, batch_size=16, shuffle=True)
    model = SimpleRNN(n_features=len(ds.features))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    for e in range(epochs):
        tot=0.0
        for x,y in dl:
            pred = model(x)
            loss = loss_fn(pred.squeeze(), y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        print(f'Epoch {e} loss {tot/len(dl):.4f}')
    torch.save(model.state_dict(), 'rnn_model.pth')

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--epochs', type=int, default=5)
    args = p.parse_args()
    train(args.data, epochs=args.epochs)
