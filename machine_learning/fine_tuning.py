from pickle import PERSID
import torch
import torch.nn as nn
import numpy as np
from machine_learning.classifier import LSTMClassifier

INPUT_DIM = 9
HIDDEN_DIM = 128

def model_reset_hidden2tag(load_model_path, before_target_dim, after_target_dim):
    model = LSTMClassifier(INPUT_DIM, HIDDEN_DIM, before_target_dim)
    model_path = load_model_path
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    #　最後の部分だけ取り換える
    model.hidden2tag = nn.Linear(HIDDEN_DIM, after_target_dim)
    return model
