import numpy as np

def policy_evaluation(env, policy, gamma, theta, max_iterations):
    value = np.zeros(env.n_states, dtype=float)

    # TODO:
    delta = np.inf 
    i = 0
    while i < max_iterations and delta >= theta:
        delta = 0
        for s in range(env.n_states):
            Vold = value[s]
            v_s = 0
            for n_s in range(env.n_states):
                v_s += env.p(n_s, s, policy[s]) * (env.r(n_s, s, policy[s]) + gamma * value[n_s])
            value[s] = v_s
            delta = np.maximum(delta, abs(Vold - value[s]))
        i += 1
    return value
    
def policy_improvement(env, value, gamma):
    policy = np.zeros(env.n_states, dtype=int)
    
    # TODO:
    q_table = np.zeros((env.n_states, env.n_actions))
    for s in range(env.n_states):
        for a in range(env.n_actions):
            v_s = 0
            for n_s in range(env.n_states):
                v_s += env.p(n_s, s, a) * (env.r(n_s, s, a) + gamma * value[n_s])
            q_table[s,a] = v_s

    policy = q_table.argmax(axis = 1)

    return policy
    
def policy_iteration(env, gamma, theta, max_iterations, policy=None):
    if policy is None:
        policy = np.zeros(env.n_states, dtype=int)
    else:
        policy = np.array(policy, dtype=int)
    
    # TODO:
    value = np.zeros(env.n_states, dtype=float)
    prev_policy = None

    counter = 0
    for i in range(max_iterations):
        value = policy_evaluation(env, policy, gamma, theta, max_iterations)
        policy = policy_improvement(env, value, gamma)

        # break when it converges
        # if (prev_policy == policy).all():
        if np.array_equal(prev_policy, policy):
            counter = i + 1
            break
        
        prev_policy = policy
    # print("Num of iterations", counter)
    return policy, value 
