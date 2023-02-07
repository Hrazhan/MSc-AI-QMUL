from time import process_time
import numpy as np
from environment.frozen_lake import FrozenLake
from model_based.policy_iteration import policy_iteration
from model_based.value_iteration import value_iteration

from model_based.policy_iteration import policy_evaluation 

def policy_value_iteration(env, gamma, theta, max_iterations):
    print('## Measuring Policy iteration for a 100 run')
    durations = []

    for i in range(100):
        start = process_time()
        policy, value = policy_iteration(env, gamma, theta, max_iterations)


        end = process_time()
        duration = end - start
        durations.append(duration)

    print("Mean time:", round(np.mean(durations), 4), "seconds")

    print("\n\n")
    print('## Measuring Value iteration for a 100 run')

    durations = []
    for i in range(100):
        start = process_time()
        policy, value = value_iteration(env, gamma, theta, max_iterations)


        end = process_time()
        duration = end - start
        durations.append(duration)

    print("Mean time:", round(np.mean(durations), 4), "seconds")
    



def sarsa_evaluation(env, max_episodes, eta, gamma, epsilon, optimal_policy, theta, seed=None):
    random_state = np.random.RandomState(seed)
    
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    q = np.zeros((env.n_states, env.n_actions))
    
    counter = 0
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
            
        if np.all(np.abs(optimal_policy - q.max(axis=1)) < theta):
            print("No of episodes:", i + 1)
            counter = i + 1
            break
        # value_temp = policy_evaluation(env, q.argmax(axis=1) ,gamma, 0.001, 100)
        # error = np.mean(np.not_equal(value_temp, optimal_policy))
        # if error <= theta:
        #     print("No of episodes:", i + 1)
        #     counter = i + 1
        #     break

    policy = q.argmax(axis=1)
    value = q.max(axis=1)
        
    return policy, value, counter



def q_learning_evaluation(env, max_episodes, eta, gamma, epsilon, optimal_policy, theta, seed=None):
    # print("## Q-Learning Evaluation")
    random_state = np.random.RandomState(seed)
    
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    q = np.zeros((env.n_states, env.n_actions))
    
    counter = 0
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
        
        if np.all(np.abs(optimal_policy - q.max(axis=1)) < theta):
            print("No of episodes:", i + 1)
            counter = i + 1
            break
        # value_temp = policy_evaluation(env, q.argmax(axis=1) ,gamma, 0.001, 100)
        # error = np.mean(np.not_equal(value_temp, optimal_policy))
        # if error <= theta:
        #     print("No of episodes:", i + 1)
        #     counter = i + 1
        #     break

    policy = q.argmax(axis=1)
    value = q.max(axis=1)
        
    return policy, value, counter


def main():
    seed = 0

    # Small lake
    lake = [['&', '.', '.', '.'],
                  ['.', '#', '.', '#'],
                  ['.', '.', '.', '#'],
                  ['#', '.', '.', '$']]

    # Big lake
    big_lake = [['&', '.', '.', '.', '.', '.', '.', '.'],
                  ['.', '.', '.', '.', '.', '.', '.', '.'],
                  ['.', '.', '.', '#', '.', '.', '.', '.'],
                  ['.', '.', '.', '.', '.', '#', '.', '.'],
                  ['.', '.', '.', '#', '.', '.', '.', '.'],
                  ['.', '#', '#', '.', '.', '.', '#', '.'],
                  ['.', '#', '.', '.', '#', '.', '#', '.'],
                  ['.', '.', '.', '#', '.', '.', '.', '$']]

    lake = np.array(big_lake)
    env = FrozenLake(lake, slip=0.1, max_steps=lake.size, seed=seed)

    gamma = 0.9
    theta = 0.001
    max_iterations = 100
    # policy_value_iteration(env, gamma, theta, max_iterations)


    max_episodes = 50000
    eta = 0.2
    epsilon = 0.1
    _, optimal_policy = policy_iteration(env, 0.9, 0.001, 100)
    theta = 0.1


    # print(optimal_policy)
    policy, value, _ = sarsa_evaluation(env, max_episodes, eta, gamma, epsilon, optimal_policy, theta, seed)
    env.render(policy, value)
    # q_learning_evaluation(env, max_episodes, eta, gamma, epsilon, optimal_policy, theta, seed)
    # simulate_sarsa(100, env, max_episodes, eta, gamma, epsilon, optimal_policy, theta, seed)
    # simulate_qlearning(100, env, max_episodes, eta, gamma, epsilon, optimal_policy, theta, seed)

def simulate_sarsa(n, env, max_episodes, eta, gamma, epsilon, optimal_policy, theta, seed):
    print("## Sarsa Evaluation")
    episodes = []
    for i in range(n):
        _, _, eps = sarsa_evaluation(env, max_episodes, eta, gamma, epsilon, optimal_policy, theta, seed)
        episodes.append(eps)
    
    print("Mean # of episodes:", round(np.mean(episodes), 4), "episodes")

def simulate_qlearning(n, env, max_episodes, eta, gamma, epsilon, optimal_policy, theta, seed):
    print("## Q Learning Evaluation")
    episodes = []
    for i in range(n):
        _, _, eps = q_learning_evaluation(env, max_episodes, eta, gamma, epsilon, optimal_policy, theta, seed)
        episodes.append(eps)
    
    print("Mean # of episodes:", round(np.mean(episodes), 4), "episodes")

if __name__ == "__main__":
    main()
