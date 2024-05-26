from DataSource import DataSource
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

def get_shape(nested_list):
    if isinstance(nested_list, list):
        return [len(nested_list)] + get_shape(nested_list[0])
    else:
        return []


trend_length = 5
cnn_length = 200
lstm_length = 60
symbol = "BTCUSDT"
start_time = "2024-03-16"
end_time = "2024-04-16"
interval = '1h'
batch_size = 32


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

class CNNModel(nn.Module):
    def __init__(self, cnn_length, cnn_embedding_length):
        super(CNNModel, self).__init__()
        self.cnn_length = cnn_length
        self.cnn_embedding_length = cnn_embedding_length

        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 5), stride=1, padding=(1, 0))
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))  
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        self.flattened_size = self.calculate_flattened_size()

        self.fc1 = nn.Linear(self.flattened_size, 256)  
        self.fc2 = nn.Linear(256, cnn_embedding_length)  

    def calculate_flattened_size(self):
        dummy_input = torch.zeros(1, 1, self.cnn_length, 5)
        dummy_output = self.pool3(self.conv3(self.pool2(self.conv2(self.pool1(self.conv1(dummy_input))))))
        return dummy_output.numel()

    def forward(self, x):
        # x: (batch_size, lstm_length, cnn_length, 5)
        batch_size, lstm_length, cnn_length, channels = x.size()

        # Reshape for CNN: (batch_size * lstm_length, 1, cnn_length, 5)
        x = x.view(batch_size * lstm_length, 1, cnn_length, channels)

        x = torch.relu(self.conv1(x))  # (batch_size * lstm_length, 16, cnn_length, 5)
        x = self.pool1(x)              # (batch_size * lstm_length, 16, cnn_length // 2, 5)
        x = torch.relu(self.conv2(x))  # (batch_size * lstm_length, 32, cnn_length // 2, 5)
        x = self.pool2(x)              # (batch_size * lstm_length, 32, cnn_length // 4, 5)
        x = torch.relu(self.conv3(x))  # (batch_size * lstm_length, 64, cnn_length // 4, 5)
        x = self.pool3(x)              # (batch_size * lstm_length, 64, cnn_length // 8, 5)

        x = x.view(batch_size * lstm_length, -1)  # (batch_size * lstm_length, flattened_size)
        x = torch.relu(self.fc1(x))               # (batch_size * lstm_length, 256)
        x = torch.relu(self.fc2(x))               # (batch_size * lstm_length, cnn_embedding_length)

        x = x.view(batch_size, lstm_length, self.cnn_embedding_length)

        return x

# cnn_embedding_length = 128

# example_input = torch.tensor(data.cnn_data[0], dtype=torch.float32)

# cnn_model = CNNModel(cnn_length, cnn_embedding_length)

# output = cnn_model(example_input)

# # Output shape
# print(output.shape)  # Should be (batch_size, lstm_length, cnn_embedding_length)
