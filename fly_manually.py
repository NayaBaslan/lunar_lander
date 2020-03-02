from __future__ import print_function

import argparse
from pyglet.window import key
import gym
import numpy as np
import pickle
import os
from datetime import datetime
import gzip
import json
from lunar_lander import LunarLander

import glob
import re


def key_press(k, mod):
    global restart
    if k==0xff0d: restart = True
    if k==key.LEFT:  a[0] = 1
    if k==key.RIGHT: a[0] = 3
    if k==key.UP:    a[0] = 2
    if k==key.DOWN:  a[0] = 0

def key_release(k, mod):
    if k==key.LEFT:  a[0] = 0
    if k==key.RIGHT: a[0] = 0
    if k==key.UP:   a[0] = 0
    if k==key.DOWN: a[0] = 0


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def store_data(data, datasets_dir="./data"):
    # save data
    file_names = glob.glob(os.path.join(datasets_dir, "*.gzip"))
    file_names.sort(key=natural_keys)
    if not file_names:
        new_index = 0
    else:
        last_index = int(re.split(r'(\d+)', file_names[-1])[1])
        new_index = last_index + 1
    if not os.path.exists(datasets_dir):
        os.mkdir(datasets_dir)
    data_file = os.path.join(datasets_dir, 'data%s.pkl.gzip'%new_index)
    f = gzip.open(data_file,'wb')
    pickle.dump(data, f)


def save_results(episode_rewards, results_dir="./results"):
    # save results
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

     # save statistics in a dictionary and write them into a .json file
    results = dict()
    results["number_episodes"] = len(episode_rewards)
    results["episode_rewards"] = episode_rewards

    results["mean_all_episodes"] = np.array(episode_rewards).mean()
    results["std_all_episodes"] = np.array(episode_rewards).std()
 
    fname = os.path.join(results_dir, "results_manually-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S"))
    fh = open(fname, "w")
    json.dump(results, fh)
    print('... finished')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--collect_data", action="store_true", default=False, help="Collect the data in a pickle file.")

    args = parser.parse_args()

    samples = {
        "state": [],
        "state_img": [],
        "next_state": [],
        "next_state_img": [],
        "reward": [],
        "action": [],
        "terminal" : [],
    }

    env = LunarLander()
    env.render()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release


    a = np.array([0])
    
    episode_rewards = []
    steps = 0
    while True:
        episode_reward = 0
        state = env.reset()
        state_img = env.render(mode="rgb_array")[::4, ::4, :]  # downsampling (every 4th pixel).

        while True:

            next_state, r, done, info = env.step(a[0])
            next_state_img = env.render(mode="rgb_array")[::4, ::4, :]
         
            episode_reward += r

            samples["state"].append(state)         # state has shape (8,)
            samples["state_img"].append(state_img)   # state_img has shape (100, 150, 3)
            samples["action"].append(np.array(a))
            samples["next_state"].append(next_state)
            samples["next_state_img"].append(next_state_img)
            samples["reward"].append(r)
            samples["terminal"].append(done)
            
            state = next_state
            steps += 1

            if steps % 3000 == 0 or done:
                print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                print("\nstep {}".format(steps))

            if args.collect_data and steps % 2500 == 0:
                print('... saving data')
                store_data(samples)
                save_results(episode_rewards)

            #env.render()
            if done: 
                print("REWARD", episode_reward)
                break
        
        episode_rewards.append(episode_reward)

    env.close()

    

   
