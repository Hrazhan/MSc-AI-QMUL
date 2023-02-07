#!/usr/bin/env python3

import numpy as np
from .environment import Environment, _printoptions
from itertools import product

class FrozenLake(Environment):
    def __init__(self, lake, slip, max_steps, seed=None):
        """
        lake: A matrix that represents the lake. For example:
         lake =  [['&', '.', '.', '.'],
                  ['.', '#', '.', '#'],
                  ['.', '.', '.', '#'],
                  ['#', '.', '.', '$']]
        slip: The probability that the agent will slip
        max_steps: The maximum number of time steps in an episode
        seed: A seed to control the random number generator (optional)
        """
        # start (&), frozen (.), hole (#), goal ($)
        self.lake = np.array(lake)
        self.lake_flat = self.lake.reshape(-1)
        
        self.slip = slip
        
        n_states = self.lake.size + 1
        n_actions = 4
        
        pi = np.zeros(n_states, dtype=float)
        pi[np.where(self.lake_flat == '&')[0]] = 1.0
        
        self.absorbing_state = n_states - 1
        
        # TODO:
        Environment.__init__(self, n_states, n_actions, max_steps, None)

        #               Up      left      down     right 
        self.actions = [(-1, 0), (0, -1), (1, 0), (0, 1)]

        self.idx_to_states = [] 
        # Indices to states (coordinates), 
        for i in range(self.lake.shape[0]):
            for j in range(self.lake.shape[1]):
                self.idx_to_states.append((i, j))
        # print(self.idx_to_states)
        # State to indicies ( we use this as lookup table to make sure the state is part of our grid)
        self.states_to_idx = {s: i for (i, s) in enumerate(self.idx_to_states)}
        # print(self.states_to_idx)
        

        # Precomputed transition probabilities
        self.transition_probs = np.zeros((self.n_states, self.n_states, self.n_actions))
        
        for state_idx, state in enumerate(self.idx_to_states):
            for action_idx, action in enumerate(self.actions):
                if self.lake[state] == '$' or self.lake[state] == '#':
                    next_state_idx = self.absorbing_state
                else:
                    next_state = (state[0] + action[0], state[1] + action[1])
                    # print("next_state", next_state)
                    # If next_state is not valid, default to current state index ( meaning if next state is outside the grid)
                    if next_state in self.states_to_idx:
                        next_state_idx = self.states_to_idx[next_state]
                    else:
                        next_state_idx = state_idx
                    # print("next_state_idx", next_state_idx)

                # However, with probability 0.1, the environment ignores the desired direction 
                # and the agent slips (moves one tile in a random direction)
                for a_idx, a in enumerate(self.actions):
                    if a == action:
                        self.transition_probs[next_state_idx, state_idx, a_idx] += 0.9 # There is 0.9 probability you move to a different tile
                    self.transition_probs[next_state_idx, state_idx, a_idx] += self.slip / (self.n_actions)
                    # print("Slip chance", self.slip / (self.n_actions))

        # Transition probabilities for the absorbing state
        for a_idx, a in enumerate(self.actions):
            self.transition_probs[self.absorbing_state, self.absorbing_state, a_idx] += 1

        # Load the precomputed transition probabilities
        # self.transition_probs = np.load('./p.npy')
        
    def step(self, action):
        state, reward, done = Environment.step(self, action)
        
        done = (state == self.absorbing_state) or done
        
        return state, reward, done
        
    def p(self, next_state, state, action):
        # TODO:
        return self.transition_probs[next_state, state, action]

    def r(self, next_state, state, action):
        # TODO:
        if state < self.n_states - 1 and self.lake_flat[state] == '$':
            return 1
        else:
            return 0

    def render(self, policy=None, value=None):
        if policy is None:
            lake = np.array(self.lake_flat)
            
            if self.state < self.absorbing_state:
                lake[self.state] = '@'
                
            print(lake.reshape(self.lake.shape))
        else:
            # UTF-8 arrows look nicer, but cannot be used in LaTeX
            # https://www.w3schools.com/charsets/ref_utf_arrows.asp
            # actions = ['^', '<', '_', '>']
            actions = ['↑', '←', '↓', '→']


            print('Lake:')
            print(self.lake)
        
            print('Policy:')
            policy = np.array([actions[a] for a in policy[:-1]])

            # We have to do add this to the end so printed policy does not overwrite the holes and goal tile with actions arrows
            policy = policy.reshape(self.lake.shape)
            for i in range(len(self.lake[0])):
                for j in range(len(self.lake[0])):
                    if self.lake[i][j] == '$':
                        policy[i][j] = '$'
                    elif self.lake[i][j] == '#':
                        policy[i][j] = '#'
                    else:
                        pass
            print(policy.reshape(self.lake.shape))
            
            print('Value:')
            with _printoptions(precision=3, suppress=True):
                print(value[:-1].reshape(self.lake.shape))
                
def play(env):
    actions = ['w', 'a', 's', 'd']
    
    state = env.reset()
    env.render()
    
    done = False
    while not done:
        c = input('\nMove: ')
        if c not in actions:
            raise Exception('Invalid action')
            
        state, r, done = env.step(actions.index(c))
        
        env.render()
        print('Reward: {0}.'.format(r))
