import numpy as np 

def q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    q = np.zeros((env.n_states, env.n_actions))
    
    for i in range(max_episodes):
        s = env.reset()
        # TODO:
        
        done = False
        while done != True:
            if np.random.random() <= epsilon[i]:
                a = np.random.randint(0, env.n_actions)
            else:
                a = np.argmax(q[s,:])

            n_s, reward, done = env.step(a)

            q[s, a] = q[s, a] + (eta[i] * (reward + (gamma * q[n_s,:].max()) - q[s, a]))
            s = n_s

    policy = q.argmax(axis=1)
    value = q.max(axis=1)
        
    return policy, value