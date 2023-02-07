from environment.frozen_lake import FrozenLake 
import numpy as np

seed = 0
# Small lake
lake =   [['&', '.', '.', '.'],
          ['.', '#', '.', '#'],
          ['.', '.', '.', '#'],
          ['#', '.', '.', '$']]

# Large lake
largeLake = [['&', '.', '.', '.', '.', '.', '.', '.'],
             ['.', '.', '.', '.', '#', '.', '.', '.'],
             ['.', '.', '.', '.', '.', '.', '#', '.'], 
             ['.', '.', '.', '.', '#', '.', '.', '.'], 
             ['.', '.', '.', '#', '.', '.', '.', '.'], 
             ['.', '#', '#', '.', '.', '.', '#', '.'], 
             ['.', '#', '.', '.', '#', '.', '#', '.'],   
             ['.', '.', '.', '#', '.', '.', '.', '$']]

lake_size = np.array(largeLake)
max_steps = lake_size.shape[0] * lake_size.shape[1] # 64
# print(max_steps)


env = FrozenLake(largeLake, slip=0.1, max_steps=max_steps, seed=seed)
done = False 
while not done:
    actions = ['w', 'a', 's', 'd']

    state = env.reset()
    env.render()

    done = False
    while not done:
        c = input('\nMove: ')
        if c not in actions:
            env.render()
            raise Exception('Invalid action')

        state, r, done = env.step(actions.index(c))
        print(done)
        env.render()
        print('Reward: {0}.'.format(r))
