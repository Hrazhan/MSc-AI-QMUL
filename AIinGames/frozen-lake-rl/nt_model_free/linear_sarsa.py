import numpy as np


def linear_sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    theta = np.zeros(env.n_features)
    
    for i in range(max_episodes):
        features = env.reset()
        
        q = features.dot(theta)

        # TODO:
        done = False
        if np.random.random() <= epsilon[i]:
            a = np.random.randint(0, env.n_actions)
        else:
            a = np.argmax(q)
        while not done:
            nxt_feat, reward, done = env.step(a)
            delta_val = reward - q[a]
            q = nxt_feat.dot(theta)
            if np.random.random() <= epsilon[i]:
                nxt_act = np.random.randint(0, env.n_actions)
            else:
                nxt_act = np.argmax(q)
            delta_val += gamma * q[nxt_act]
            theta += eta[i] * delta_val * features[a]
            features = nxt_feat
            a = nxt_act
    return theta
