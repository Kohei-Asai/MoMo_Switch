import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import math
import pandas as pd
import numpy as np
import csv


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
    x = x.to(device)
    y = torch.tensor(y, dtype=torch.long)
    y = y.to(device)
    data.append((x, y))
  return data

def make_tensors_from_csv(load_paths):
  data = []
  for load_path in load_paths:
    with open(load_path, 'r') as f:
      reader = csv.reader(f)
      data_rows = []
      for row in reader:
        data_one_row = []
        for item in row:
          data_one_row.append(float(item))
        data_rows.append(data_one_row)
    x = torch.t(torch.tensor(data_rows[:-1], dtype=torch.float))
    x = x.to(device)
    y = torch.tensor(data_rows[-1], dtype=torch.long)
    y = y.to(device)
    data.append((x, y))
  return data

index2category = {0:'Brush', 1:'Drink', 2:'WashFace', 3:'Walk', 4:'Senobi'}
category2index = {'Brush':0, "Drink":1, "WashFace":2, "Walk":3, "Senobi":4}
def category2tensor(cat):
    return torch.tensor([category2index[cat]], dtype=torch.long)

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




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#mat
#train_data = make_tensors_from_mat(['data/data_1.mat', 'data/data_2.mat', 'data/data_3.mat', 'data/data_4.mat'])

#csv
make_tensors_from_csv(['kari.csv'])

INPUT_DIM = 9
HIDDEN_DIM = 128
TARGET_DIM = 5
model = LSTMClassifier(INPUT_DIM, HIDDEN_DIM, TARGET_DIM)

#学習済みモデルを使いたい場合
#model_path = 'model.pth'
#model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

model.to(device)

from sklearn.model_selection import train_test_split
import torch.optim as optim

loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters())

losses = []

NUM_EPOCHS = 50

for epoch in range(NUM_EPOCHS):
  all_loss = 0
  for inputs, answer in train_data:
    model.zero_grad()
    out = model(inputs)
    loss = loss_function(torch.unsqueeze(out, 0), torch.unsqueeze(answer, 0))
    loss.backward()
    optimizer.step()
    all_loss += loss.item()
  losses.append(all_loss)
  print("epoch", epoch, "\t", "loss", all_loss)
print("done.")

save_path = '/content/model.pth'
torch.save(model.state_dict(), save_path)


def classificate(model, x, threshold=-float('inf')):
  y = model.forward(x)
  y_added = torch.vstack([y, torch.tensor([threshold] * y.shape[1]).to(device)])
  classified = torch.argmax(y_added, dim=0)
  return classified

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

def compare_accuracy(answer_tensor, prediction_tensor):
  length = answer_tensor.shape[0]
  correct = 0
  for i in range(length):
    if answer_tensor[i].item() == prediction_tensor[i].item():
      correct += 1
  return correct / length

test_x, test_y = make_tensors_from_mat(['data/data_5.mat'])[0]
predicted_y = classificate(model, test_x, -0.15)
print(compare_accuracy(test_y, predicted_y))
compare_graph(test_y, predicted_y)
