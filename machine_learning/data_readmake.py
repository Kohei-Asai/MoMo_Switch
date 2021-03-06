import torch
import numpy as np
import csv
import glob
import os
import csv
import random

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

def make_csvfile_onecategory(x, category_index, save_path):
    #xは縦が時間方向、横が自由度方向、配列かnumpyの型を受け付ける
    #save_pathは.csvまで含める
    if type(x) != np.ndarray:
        x = np.array(x)
    index_np = np.array([[category_index] * x.shape[0]]).T
    all_np = np.hstack([x, index_np])
    all_list = all_np.tolist()
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(all_list)

def make_merged_csvfiles(person_directory, shuffle=True):
    categories = []
    for name in glob.glob(person_directory + '\*'):
        categories.append(name[len(person_directory) + 1:])
    print(categories)

    if not os.path.exists(person_directory + '/Merged'):
      os.mkdir(person_directory + '/Merged')
    i=1
    flag = True
    while flag:
        data = ""
        if shuffle:
          now_categories = random.sample(categories, len(categories))
        else:
          now_categories = categories
        for category in now_categories:
            csv_path = person_directory + '/'+ category + '/'+  category + str(i) + '.csv'
            if os.path.exists(csv_path):
                f = open(csv_path)
                if flag:
                    data += f.read()
                f.close()
            else:
                flag = False
        if flag:
            with open(person_directory + '/Merged/' + str(i) + '.csv' , 'w') as f:
                f.write(data)
        i += 1

def make_tensors_from_csv_test():
    answer = make_tensors_from_csv(['data/person1_test/Drink/1.csv', 'data/person1_test/Drink/2.csv'])
    print(answer)

def make_csvfile_onecategory_test():
    make_csvfile_onecategory([[1, 2], [3, 4], [5, 6]], 1, 'data/person1_test/Drink/2.csv')

if __name__ == "__main__":
    make_merged_csvfiles('data/hirata')