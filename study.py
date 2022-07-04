import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

def index2category(index):
  this_dic = {0:'Brush', 1:'Drink', 2:'WashFace', 3:'Walk', 4:'Senobi'}
  return this_dic[index]

def category2index(cat):
  this_dic = {'Brush':0, "Drink":1, "WashFace":2, "Walk":3, "Senobi":4}
  return this_dic[cat]

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



def get_device():
  return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#mat
#train_data = make_tensors_from_mat(['data/data_1.mat', 'data/data_2.mat', 'data/data_3.mat', 'data/data_4.mat'])

#csv(['kari.csv'])

def get_new_model(input_dim=9, hidden_dim=128, target_dim=5, device='cpu'):
  model = LSTMClassifier(input_dim, hidden_dim, target_dim)
  model.to(device)
  return model

def load_model(model_path, input_dim=9, hidden_dim=128, target_dim=5, device='cpu'):
  model = LSTMClassifier(input_dim, hidden_dim, target_dim)
  model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
  model.to(device)
  return model

def study(model, train_data, num_epochs=50):
  loss_function = nn.NLLLoss()
  optimizer = optim.Adam(model.parameters())
  losses = []
  for epoch in range(num_epochs):
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
  return model

def save_model(model, save_path):
  #save_path = '/content/model.pth'
  torch.save(model.state_dict(), save_path)


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
  plt.plot(t, answer_np)
  plt.plot(t, prediction_np)

def compare_accuracy(answer_tensor, prediction_tensor):
  length = answer_tensor.shape[0]
  correct = 0
  for i in range(length):
    if answer_tensor[i].item() == prediction_tensor[i].item():
      correct += 1
  return correct / length

#test_x, test_y = make_tensors_from_mat(['data/data_5.mat'])[0]
#predicted_y = classificate(model, test_x, -0.15)
#print(compare_accuracy(test_y, predicted_y))
#compare_graph(test_y, predicted_y)
