import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import math
import pandas as pd
import numpy as np

index2category = {0:'Brush', 1:'Drink', 2:'WashFace', 3:'Walk', 4:'Senobi'}
category2index = {'Brush':0, "Drink":1, "WashFace":2, "Walk":3, "Senobi":4}
def category2tensor(cat):
    return torch.tensor([category2index[cat]], dtype=torch.long)

def mat2array(load_path):
    import scipy.io
    mat = scipy.io.loadmat(load_path)
    x = mat['x']
    y = mat['y']
    y = y[0]
    return (x, y)
def make_tensors_from_mat(load_paths):
    data = []
    for load_path in load_paths:
        x, y = mat2array(load_path)
        x = torch.t(torch.tensor(x, dtype=torch.float))
        y = torch.tensor(y, dtype=torch.long)
        data.append((x, y))
    return data

test_x, test_y = make_tensors_from_mat(['data_5.mat'])[0]

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, target_dim):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, target_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sequence):
        out, _ = self.lstm(sequence)
        out = self.hidden2tag(out)
        scores = self.softmax(out)
        scores = torch.t(scores)
        return scores

INPUT_DIM = 9
HIDDEN_DIM = 128
TARGET_DIM = 5
model = LSTMClassifier(INPUT_DIM, HIDDEN_DIM, TARGET_DIM)
#学習済みモデルを使いたい場合
model_path = 'model_from_matfile_9freedom_1.pth'
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

def classificate(model, x, threshold=-float('inf')):
    y = model.forward(x)
    y_added = torch.vstack([y, torch.tensor([threshold] * y.shape[1])])
    classified = torch.argmax(y_added, dim=0)
    return classified


#ここからリアルタイムでやる
predicted_y = classificate(model, test_x, -0.15)
print(predicted_y)

def compare_graph(answer_tensor, prediction_tensor):
    import matplotlib.pyplot as plt
    import torch
    import numpy as np
    HZ = 100
    answer_np = answer_tensor.to('cpu').detach().numpy().copy()
    prediction_np = prediction_tensor.to('cpu').detach().numpy().copy()
    t = np.arange(0, answer_np.shape[0]/HZ, 1/HZ)
    plt.plot(t, answer_np)
    plt.plot(t, prediction_np)
    plt.show()
compare_graph(test_y, predicted_y)