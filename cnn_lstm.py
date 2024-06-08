from DataSource import DataSource
from cnn import CNNModel
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

trend_length = 5
cnn_length = 200
lstm_length = 60
symbol = "BTCUSDT"
start_time = "2024-01-16"
end_time = "2024-04-16"
interval = '1h'
batch_size = 32
ohclv_length = 5  # open, high, low, close, volume의 길이
num_lstm_layers = 3    # lstm layer의 개수
cnn_embedding_length = 16

# data = DataSource(
#     symbol=symbol,
#     start_time=start_time,
#     end_time=end_time,
#     interval=interval,
#     trend_length=trend_length,
#     cnn_length=cnn_length,
#     lstm_length=lstm_length,
#     batch_size=batch_size
# )

class CNNLSTMModel(nn.Module):
    def __init__(self, cnn_length, cnn_embedding_length, ohclv_length, lstm_hidden_dim, output_dim, num_lstm_layers):
        super(CNNLSTMModel, self).__init__()
        self.ohclv_length = ohclv_length
        self.cnn_model = CNNModel(cnn_length, cnn_embedding_length)
        self.lstm = nn.LSTM(input_size=cnn_embedding_length+ohclv_length, hidden_size=lstm_hidden_dim, num_layers=num_lstm_layers, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_dim, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

    def forward(self, x, ohclv):
        batch_size, lstm_length, cnn_length, channels = x.size()

        cnn_out = self.cnn_model(x)  # (batch_size, lstm_length, cnn_embedding_length)
        
        # ohclv: (batch_size, lstm_length, ohclv_length)
        # ohclv_expanded = ohclv.unsqueeze(1).expand(-1, lstm_length, -1)  # (batch_size, lstm_length, ohclv_length)
        
        combined_input = torch.cat((cnn_out, ohclv), dim=2)  # (batch_size, lstm_length, cnn_embedding_length + ohclv_length)
        
        lstm_out, _ = self.lstm(combined_input)  # (batch_size, lstm_length, lstm_hidden_dim)
        lstm_out = lstm_out[:, -1, :]
        
        output = self.fc(lstm_out)  # (batch_size, output_dim)
        output = self.dropout(output)
        output = self.fc1(self.relu(output))


        return output

# lstm_hidden_dim = 64
# output_dim = 1

# example_input = torch.tensor(data.cnn_data[0], dtype=torch.float32)
# example_ohclv = torch.tensor(data.np_cnn_data[:len(data.cnn_data[0]), -ohclv_length:], dtype=torch.float32)

# cnnlstm_model = CNNLSTMModel(cnn_length, cnn_embedding_length, ohclv_length, lstm_hidden_dim, output_dim, num_lstm_layers)

# output = cnnlstm_model(example_input, example_ohclv)

# print("CNN_LSTM output shape:", output.shape)