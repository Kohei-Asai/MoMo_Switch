import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import math
import pandas as pd
import numpy as np
import csv
import torch.optim as optim
import matplotlib.pyplot as plt
import glob
import os
import csv

def mat2array(load_path):
  import scipy.io
  mat = scipy.io.loadmat(load_path)
  x = mat['x']
  y = mat['y']
  y = y[0]
  return (x, y)

def make_tensors_from_mat(load_paths, device='cpu'):
  data = []
  for load_path in load_paths:
    x, y = mat2array(load_path)
    x = torch.t(torch.tensor(x, dtype=torch.float))
    x = x.to(device)
    y = torch.tensor(y, dtype=torch.long)
    y = y.to(device)
    data.append((x, y))
  return data

def make_tensors_from_csv(load_paths, device='cpu'):
  #csvファイルは、縦軸が時間方向、横軸が自由度＋右端はカテゴリー番号
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
    all_matrix = torch.tensor(data_rows, dtype=torch.float)
    x = all_matrix[:, :-1]
    x = x.to(device)
    y = all_matrix[:, -1]
    y = y.to(device)
    data.append((x, y))
  return data

def make_merged_csvfiles(person_directory):
    categories = []
    for name in glob.glob(person_directory + '\*'):
        categories.append(name[len(person_directory) + 1:])
    
    os.mkdir(person_directory + '/Merged')
    i=1
    flag = True
    while flag:
        data = ""
        for category in categories:
            try:
                f = open(person_directory + '/'+ category + '/'+  str(i) + '.csv')
            except OSError as e:
                flag = False
            else:
                if flag:
                    data += f.read()
                f.close()
        if flag:
            with open(person_directory + '/Merged/' + str(i) + '.csv' , 'w') as f:
                f.write(data)
        i += 1