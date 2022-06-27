import torch
import torch.nn as nn
import scipy.io
import matplotlib.pyplot as plt
import numpy as np

index2category = {0:'Brush', 1:'Drink', 2:'WashFace', 3:'Walk', 4:'Senobi'}
category2index = {'Brush':0, "Drink":1, "WashFace":2, "Walk":3, "Senobi":4}

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

def mat2array(load_path):
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

def load_model(model_path, input_dim,  hidden_dim, target_dim):
    model = LSTMClassifier(input_dim, hidden_dim, target_dim)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model

def classificate(model, x, threshold=-float('inf')):
    y = model.forward(x)
    y_added = torch.vstack([y, torch.tensor([threshold] * y.shape[1])])
    classified = torch.argmax(y_added, dim=0)
    return classified

def compare_graph(answer_tensor, prediction_tensor):
    HZ = 100
    answer_np = answer_tensor.to('cpu').detach().numpy().copy()
    prediction_np = prediction_tensor.to('cpu').detach().numpy().copy()
    t = np.arange(0, answer_np.shape[0]/HZ, 1/HZ)
    plt.plot(t, answer_np, label="answer")
    plt.plot(t, prediction_np, label="predicted")
    plt.xlabel('time')
    plt.ylabel('action')
    plt.legend()
    plt.show()