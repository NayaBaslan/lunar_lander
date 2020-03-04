import numpy as np


IDLE = 0
LEFT = 1
UP = 2
RIGHT = 3 

def one_hot(labels):
    """
    this creates a one hot encoding from a flat vector:
    i.e. given y = [0,2,1]
     it creates y_one_hot = [[1,0,0], [0,0,1], [0,1,0]]
    """
    classes = np.unique(labels)
    n_classes = classes.size
    one_hot_labels = np.zeros(labels.shape + (n_classes,))
    for c in classes:
        one_hot_labels[labels == c, c] = 1.0
    return one_hot_labels

def rgb2gray(rgb):
    """ 
    this method converts rgb images to grayscale arrays and normalizes the images.
    """
    gray = np.dot(rgb[...,:3], [0.2125, 0.7154, 0.0721])
    gray =  2 * gray.astype('float32') - 1 
    return gray 


def preprocessing(X_train, y_train, X_valid, y_valid):
    print("preprocess states")

    y_train = y_train.astype(np.int).flatten()
    y_valid = y_valid.astype(np.int).flatten()

    print("#samples with action=IDLE : ", len(y_train[y_train == IDLE]) )
    print("#samples with action=UP : ", len(y_train[y_train == UP]) )
    print("#samples with action=LEFT    : ", len(y_train[y_train == LEFT]) )
    print("#samples with action=RIGHT     : ", len(y_train[y_train == RIGHT]))

    return np.array([rgb2gray(img) for img in X_train]).reshape(-1, 100, 150, 1), y_train, one_hot(y_train), \
            np.array([rgb2gray(img) for img in X_valid]).reshape(-1, 100, 150, 1), y_valid, one_hot(y_valid)


class EpisodeStats:
    """
    This class tracks statistics like episode reward or action usage.
    """
    def __init__(self):
        self.episode_reward = 0
        self.actions_ids = []

    def step(self, reward, action_id):
        self.episode_reward += reward
        self.actions_ids.append(action_id)

    def get_action_usage(self, action_id):
        ids = np.array(self.actions_ids)
        return (len(ids[ids == action_id]) / len(ids))
