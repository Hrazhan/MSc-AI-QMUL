from baselines.common.atari_wrappers import make_atari, wrap_deepmind
import numpy as np
import tensorflow as tf
from tensorflow import keras
import gym
import random

seed = 42

model = keras.models.load_model('./game_ai/model')

env = make_atari("BreakoutNoFrameskip-v4")
env = wrap_deepmind(env, frame_stack=True, scale=True)
env.seed(seed)

env = gym.wrappers.Monitor(env, './game_ai/videos',
                          video_callable=lambda episode_id: True, force=True)

n_episodes = 10
returns = []

# print(env.action_space.n)
                       
for _ in range(n_episodes):
    ret = 0
    state = np.array(env.reset())
    
    done = False
    
    while not done:
        

        # Fix me
        state_tensor = tf.convert_to_tensor(state)
        state_tensor = tf.expand_dims(state_tensor, 0)
        # predict quality of action using the trained model
        action_probs = model(state_tensor, training=False)
        # select the best action
        action = tf.argmax(action_probs[0]).numpy()

        #apply selected action to the enviroment
        state_next, reward, done, _ = env.step(action)
        state_next = np.array(state_next)
        state = state_next
        ret += reward
        
    returns.append(ret)
    
env.close()

print('Returns: {}'.format(returns))