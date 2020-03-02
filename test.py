from __future__ import print_function

from datetime import datetime
import numpy as np
import gym
import os
import json

from agent.bc_agent import BCAgent
from utils import *
from lunar_lander import LunarLander
from utils import rgb2gray

from config import Config

import torch


def run_episode(env, agent, config, rendering=True, max_timesteps=10000):
    
    episode_reward = 0
    step = 0

    state = env.reset()
    count=0
    state_image_hist=[]
    history_length=conf.history_length
    while count <history_length:
        count+=1
        state_img = env.render(mode="rgb_array")[::4, ::4, :]  # downsampling (every 4th pixel).
        state_image_hist.append(np.array([rgb2gray(img) for img in state_img]).reshape(1,100, 150)[0])
        
    state_image_hist=np.array(state_image_hist).astype('float32')    
    #print(state_image_hist.shape)
    # fix bug of curropted states without rendering in gym environments
    env.viewer.window.dispatch_events() 
    
    while True:
        
        # TODO: preprocess the state in the same way than in your preprocessing in train_agent.py
        
        
        
        with torch.no_grad():
            a = int(torch.argmax(agent.predict(torch.tensor(state_image_hist).view(-1,history_length,100,150))))
        next_state, r, done, info = env.step(a)   
        next_state_img = env.render(mode="rgb_array")[::4, ::4, :]
        episode_reward += r       
        state = next_state
        state_image_tmp=np.zeros_like(state_image_hist)
        state_image_tmp[0:history_length-1]=state_image_hist[1:history_length]
        
        state_image_tmp[-1]=np.array([rgb2gray(img) for img in next_state_img]).reshape(1,100, 150)[0]
        
        state_image_hist=state_image_tmp
        #state_img=next_state_img
        step += 1
        
        if rendering:
            env.render()

        if done or step > max_timesteps: 
            break

    return episode_reward


if __name__ == "__main__":

    # important: probably it doesn't work for you to set rendering to False for evaluation
    rendering = True                     
    
    conf = Config()

    # TODO: load agent
    agent = BCAgent(conf.agent_type,conf.lr,conf.hidden_units,conf.history_length)
    agent.load("models/20200229-232502_bc_agent.pt")

    env = LunarLander()

    episode_rewards = []
    for i in range(conf.n_test_episodes):
        episode_reward = run_episode(env, agent, conf, rendering=rendering)
        print(episode_reward)
        episode_rewards.append(episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()
 
    fname = "results/results_bc_agent-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = open(fname, "w")
    json.dump(results, fh)
            
    env.close()
    print('... finished')
