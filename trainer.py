
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from glob import glob, iglob
import torch.nn as nn
import yfinance as yf
import pandas as pd
import numpy as np
import torch

class ForeCastLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def scale_data(df):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df.values)
    return scaled, scaler

def create_sequences(data, seq_length=50):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i : i + seq_length])
        y.append(data[i + seq_length])
    return np.array(x), np.array(y)

model = ForeCastLSTM()

all_files = glob('path/to/reports/**/*.csv', recursive=True)
all_files.sort()

df = pd.concat([pd.read_csv(i, low_memory=False)[1:][['Usage type', 'Total costs($)']] for i in all_files])

scaled_data, scaler = scale_data(df[["Total costs($)"]])

df["ScaledClose"] = scaled_data

X_np, y_np = create_sequences(scaled_data, seq_length=1)


X_tensor = torch.tensor(X_np, dtype=torch.float32)
y_tensor = torch.tensor(y_np, dtype=torch.float32)

dataset = TensorDataset(X_tensor, y_tensor)

loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = ForeCastLSTM(input_size=1, hidden_size=64, num_layers=2)

loss_fn = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

loss_history = []
for epoch in range(10):
    epoch_loss = 0.0
    for X_batch, y_batch in loader:
        output = model(X_batch)
        loss = loss_fn(output, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(loader)
    loss_history.append(avg_loss)


with torch.no_grad():
    last_seq = torch.tensor(scaled_data[-50:], dtype=torch.float32).unsqueeze(0)
    predicted = model(last_seq).numpy()
    predicted_price = scaler.inverse_transform(predicted)[0][0]


print({
        "loss_history": [float(l) for l in loss_history],
        "predicted_next_price": float(predicted_price)
    })

