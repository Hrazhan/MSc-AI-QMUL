import numpy as np

def value_iteration(env, gamma, theta, max_iterations, value=None):
    if value is None:
        value = np.zeros(env.n_states)
    else:
        value = np.array(value, dtype=float)

    # TODO:
    policy = np.zeros(env.n_states, dtype=int)

    counter = 0
    for i in range(max_iterations):
        delta = 0

        for s in range(env.n_states):
            Vold = value[s]
            action_values_arr = [0] * env.n_actions
            for a in range(env.n_actions):
                for n_s in range(env.n_states):
                    action_values_arr[a] += (env.p(n_s, s, a) * (env.r(n_s, s, a) + gamma * value[n_s]))

            value[s] = np.max(action_values_arr)

            delta = np.maximum(delta, abs(Vold - value[s]))

        # the tolerance parameter determining the accuracy of the estimation
        if delta < theta:
            counter = i + 1
            break

    # print("Num of iterations", counter)
    
    q_table = np.zeros((env.n_states, env.n_actions))
    for s in range(env.n_states):
        for a in range(env.n_actions):
            v_s = 0
            for n_s in range(env.n_states):
                v_s += env.p(n_s, s, a) * (env.r(n_s, s, a) + gamma * value[n_s])
            q_table[s,a] = v_s

    policy = np.argmax(q_table, axis=1)

    return policy, value 


