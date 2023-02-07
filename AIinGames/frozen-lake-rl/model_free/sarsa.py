import numpy as np

def sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    q = np.zeros((env.n_states, env.n_actions))
    
    for i in range(max_episodes):
        s = env.reset()
        # TODO:
    
        # select an action based on E-greedy 
        if np.random.random() <= epsilon[i]:
            a = np.random.randint(0, env.n_actions)
        else:
            a = np.argmax(q[s,:])

        done = False
        while not done:
            n_s, reward, done = env.step(a)

            # Select new action using e greedy for next state
            if np.random.random() <= epsilon[i]:
                next_action = np.random.randint(0, env.n_actions)
            else:
                next_action = np.argmax(q[n_s,:])

            q[s, a] = q[s, a] + (eta[i] * (reward + (gamma * q[n_s, next_action]) - q[s, a]))
            s = n_s
            a = next_action

    policy = q.argmax(axis=1)
    value = q.max(axis=1)
        
    return policy, value
