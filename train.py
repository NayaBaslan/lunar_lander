from __future__ import print_function

import os
import pickle
import gzip
import matplotlib.pyplot as plt
import random as rnd
from agent.bc_agent import BCAgent
from utils import rgb2gray
from datetime import datetime

import numpy as np
from tensorboard_evaluation import Evaluation

import torch
import glob

from config import Config

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def read_data(datasets_dir="./data", frac=0.1):
    print("... read data")
    file_names = glob.glob(os.path.join(datasets_dir, "*.gzip"))
    print(file_names)
    X = []
    y = []
    X_tr = []
    y_tr = []
    n_samples = 0
    for data_file in file_names:
        f = gzip.open(data_file, 'rb')
        data = pickle.load(f)
        n_samples += len(data["state"])
        X.extend(data["state_img"])  # Hint: to access images use state_img here!
        y.extend(data["action"])

    X = np.array(X).astype('float32')
    y = np.array(y).astype('float32')
    # split data into training and validation set
    X_train, y_train = X[:int((1 - frac) * n_samples)], y[:int((1 - frac) * n_samples)]
    X_valid, y_valid = X[int((1 - frac) * n_samples):], y[int((1 - frac) * n_samples):]

    return X_train, y_train, X_valid, y_valid


def onehot(y_train):
    y_train_temp = np.zeros((y_train.shape[0], 4))
    for idx in range(len(y_train)):
        y_train_temp[idx] = np.eye(4)[int(y_train[idx][0])]

    return y_train_temp


def preprocessing(X_train, y_train, X_valid, y_valid, conf):
    # TODO: preprocess your data here. For the images:

    # 1. convert the images in X_train/X_valid to gray scale. If you use rgb2gray() from utils.py, the output shape (100, 150, 1)
    X_train = np.array([rgb2gray(img).reshape(1, 100, 150) for img in X_train])
    X_valid = np.array([rgb2gray(img).reshape(1, 100, 150) for img in X_valid])
    # History:
    # At first you should only use the current image as input to your network to learn the next action. Then the input states
    # have shape (100, 150, 1). Later, add a history of the last N images to your state so that a state has shape (100, 150, N).
    skip_frames, history_length = conf.skip_frames, conf.history_length

    X_tr, y_tr = skip_history(X_train, y_train, history_length, skip_frames)
    X_val, y_val = skip_history(X_valid, y_valid, history_length, skip_frames)
    # Hint: you can also implement frame skipping
    return X_tr, y_tr, X_val, y_val


def train_model(X_train, y_train, X_valid, y_valid, config, model_dir="./models", tensorboard_dir="./tensorboard"):
    n_minibatches, batch_size, lr, hidden_units, history_length, agent_type = config.n_minibatches, config.batch_size, config.lr, config.hidden_units, config.history_length, config.agent_type

    # create result and model folders
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    print("... train model")

    # TODO: specify your agent with the neural network in agents/bc_agent.py
    agent = BCAgent(agent_type, lr, hidden_units, history_length)
    # states=['training_accuracy']
    states = ['training_accuracy', 'validation_accuracy', 'loss']

    tensorboard_eval = Evaluation(tensorboard_dir,"Learning_curves",states)

    # TODO: implement the training

    # 1. write a method sample_minibatch and perform an update step
    def sample_minibatch(X, y):
        length = X.shape[0]
        X_batch = []
        y_batch = []
        sampled_idx = rnd.sample(range(1, length), batch_size)

        for idx in sampled_idx:
            X_batch.append(X[idx])
            y_batch.append(y[idx])
        return X_batch, y_batch

    # 2. compute training/ validation accuracy and loss for the batch and visualize them with tensorboard. You can watch the progress of
    #    your training *during* the training in your web browser
    # 
    # training loop
    for i in range(n_minibatches):
        X_batch, y_batch = sample_minibatch(X_train, y_train)
        loss = agent.update(X_batch, y_batch)
        if i % 10 == 0:
            training_correct = 0
            training_total = 0
            validation_correct = 0
            validation_total = 0
            training_accuracy = 0
            validation_accuracy = 0
            with torch.no_grad():
                output_training = agent.predict(torch.tensor(X_batch).to(device))
                torch.cuda.empty_cache()
                output_validation = agent.predict(torch.tensor(X_valid).to(device))
            for idx, label in enumerate(output_training):
                if torch.argmax(label) == torch.tensor(y_batch[idx], dtype=torch.long, device=device):
                    training_correct += 1
                training_total += 1
            for idx, label in enumerate(output_validation):
                if torch.argmax(label) == torch.tensor(y_valid[idx], dtype=torch.long, device=device):
                    validation_correct += 1
                validation_total += 1

            training_accuracy = training_correct / training_total
            validation_accuracy = validation_correct / validation_total
            print("Episode %d of %d" % (i, n_minibatches))
            print("Training accuracy: %f" % training_accuracy)

            print("Validation accuracy: %f" % validation_accuracy)
            # compute training/ validation accuracy and write it to tensorboard
            eval_dic = {'training_accuracy': training_accuracy, 'validation_accuracy': validation_accuracy,
                        'loss': loss.item()}
            # eval_dic={'training_accuracy':training_accuracy}
            tensorboard_eval.write_episode_data(i,eval_dic)

    # save your agent
    save_file = datetime.now().strftime("%Y%m%d-%H%M%S") + "_bc_agent.pt"
    model_dir = agent.save(os.path.join(model_dir, save_file))
    print("Model saved in file: %s" % model_dir)


def skip_history(X, y, hist_len, n_skip):
    length = len(X)
    idx = 0
    X_seq = []
    y_seq = []
    while idx + n_skip + 1 < length:
        X_tmp = []
        s = (1, 100, 150)
        y_tmp = np.zeros(s)
        while len(X_tmp) < hist_len:
            X_tmp.append(X[idx][0])
            y_tmp = y[idx]
            if idx + n_skip + 1 < length:
                idx = idx + n_skip + 1
            else:
                X_tmp = []
                break
        if len(X_tmp) != 0:
            X_tmp = np.array(X_tmp).astype('float32').reshape(hist_len, 100, 150)
            X_seq.append(X_tmp)
            y_seq.append(y_tmp)

    X_seq = np.array(X_seq).astype('float32')
    y_seq = np.array(y_seq).astype('float32')
    return X_seq, y_seq


if __name__ == "__main__":

    # read data    
    X_train, y_train, X_valid, y_valid = read_data("./data")

    conf = Config()

    # preprocess data
    if conf.agent_type == "ResNet":
        X_train, y_train, X_valid, y_valid = preprocessing(X_train, y_train, X_valid, y_valid, conf)

    # train model (you can change the parameters!)
    train_model(X_train, y_train, X_valid, y_valid, conf)
