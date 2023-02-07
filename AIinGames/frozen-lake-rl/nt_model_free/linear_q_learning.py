import numpy as np


def linear_q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    theta = np.zeros(env.n_features)

    for i in range(max_episodes):
        features = env.reset()


        # TODO:
        q = features.dot(theta)
        done = False
        while not done:
            if np.random.random() <= epsilon[i]:
                a = np.random.randint(0, env.n_actions)
            else:
                a = np.argmax(q)
            nxt_feat, reward, done = env.step(a)
            delta_val = reward - q[a]
            q = nxt_feat.dot(theta)
            delta_val += gamma * max(q)
            theta += eta[i] * delta_val * features[a]
            features = nxt_feat

    return theta
