from pickle import PERSID
import torch
import torch.nn as nn
import numpy as np
from classifier import LSTMClassifier
import glob
import csv
import os

INPUT_DIM = 9
HIDDEN_DIM = 128

def model_reset_hidden2tag(load_model_path, before_target_dim, after_target_dim):
    model = LSTMClassifier(INPUT_DIM, HIDDEN_DIM, before_target_dim)
    model_path = load_model_path
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    #　最後の部分だけ取り換える
    model.hidden2tag = nn.Linear(HIDDEN_DIM, after_target_dim)
    return model

def save_merged_csvfiles(person_directory):
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

            
                

def main():
    save_merged_csvfiles('.\data\person1')

if __name__ == "__main__":
    main()
