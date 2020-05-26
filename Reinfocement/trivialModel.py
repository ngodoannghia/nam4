import numpy as np
from copy import deepcopy
from env import env
import matplotlib.pyplot as plt
import pandas as pd


class TrivialModel:
    def __init__(self, rand = np.random):
        self.model = dict()
        self.rand = rand
    
    def feed(self, state, action, next_state, reward):
        state = deepcopy(state)
        next_state = deepcopy(next_state)
        if tuple(state) not in self.model.keys():
            self.model[tuple(state)] = dict()
        self.model[tuple(state)][action] = [list(next_state), reward]
    
    def sample(self):
        state_index = self.rand.choice(range(len(self.model.keys())))
        state = list(self.model)[state_index]
        action_index = self.rand.choice(range(len(self.model[state].keys())))
        action = list(self.model[state])[action_index]
        next_state, reward = self.model[state][action]
        state = deepcopy(state)
        next_state = deepcopy(next_state)

        return list(state), action, list(next_state), reward
class DynaParamas:
    def __init__(self):
        self.gamma = 0.95

        self.epsilon = 0.1

        self.planning_steps = 5

        self.alpha = 0.1

        self.runs = 10

        self.theta = 0

def choose_action(q_value, e, dyna_params):
    if np.random.binomial(1, dyna_params.epsilon) == 1:
        a = np.random.choice(e.action)
        return a
    else:
        if tuple(e.state) not in q_value.keys():
            return np.random.choice(e.action)
        else:            
            values = q_value[tuple(e.state)]
            max = -1000000
            action = 0
            for tmp in values.keys():
                if values[tmp] > max:
                    max = values[tmp]
                    action = tmp
            return action

def getMax(q_value,state):
    max = -10000
    values = q_value[tuple(state)]
    for tmp in values.keys():
        if values[tmp] > max:
            max = values[tmp]
    
    return max
        

def dyna_q(q_value, model, e, dyna_params, max_step):

    e.state = e.init_state()

  
    step = 0

    list_reward = []
    avg_list_reward = []

    while True:
        
        action = choose_action(q_value, e, dyna_params)
        next_state, reward, done, _ = e.step(action)

        list_reward.append(e.reward_time + e.reward_bak + e.reward_bat)
        avg_list_reward.append(np.mean(list_reward[:]))

        if tuple(e.state) not in q_value.keys():
            q_value[tuple(e.state)] = {}
            q_value[tuple(e.state)][action] = 0
            if tuple(next_state) not in q_value.keys():
                q_value[tuple(next_state)] = {}
                q_value[tuple(next_state)][action] = 0
                q_value[tuple(e.state)][action] += dyna_params.alpha *(
                    reward + dyna_params.gamma*getMax(q_value, next_state) - q_value[tuple(e.state)][action])
            else:
                q_value[tuple(e.state)][action] += dyna_params.alpha *(
                    reward + dyna_params.gamma*getMax(q_value, next_state) - q_value[tuple(e.state)][action])
        else:
            if tuple(next_state) not in q_value.keys():
                q_value[tuple(next_state)] = {}
                q_value[tuple(next_state)][action] = 0
                q_value[tuple(e.state)][action] += dyna_params.alpha *(
                    reward + dyna_params.gamma*getMax(q_value, next_state) - q_value[tuple(e.state)][action])
            else:
                q_value[tuple(e.state)][action] += dyna_params.alpha *(
                    reward + dyna_params.gamma*getMax(q_value, next_state) - q_value[tuple(e.state)][action])

        model.feed(e.state, action, next_state, reward)

        for t in range(0, dyna_params.planning_steps):
            state_, action_, next_state_ , reward_ = model.sample()
            if tuple(state_) not in q_value.keys():
                q_value[tuple(state_)] = {}
                q_value[tuple(state_)][action_] = 0
                if tuple(next_state_) not in q_value.keys():
                    q_value[tuple(next_state_)] = {}
                    q_value[tuple(next_state_)][action_] = 0
                    q_value[tuple(state_)][action_] += dyna_params.alpha *(
                        reward + dyna_params.gamma*getMax(q_value, next_state_) - q_value[tuple(state_)][action_])
                else:
                    q_value[tuple(state_)][action_] += dyna_params.alpha *(
                        reward + dyna_params.gamma*getMax(q_value, next_state_) - q_value[tuple(state_)][action_])
            else:
                if tuple(next_state_) not in q_value.keys():
                    q_value[tuple(next_state_)] = {}
                    q_value[tuple(next_state_)][action_] = 0
                    q_value[tuple(state_)][action_] += dyna_params.alpha *(
                        reward + dyna_params.gamma*getMax(q_value, next_state_) - q_value[tuple(state_)][action_])
                else:
                    q_value[tuple(state_)][action_] += dyna_params.alpha *(
                        reward + dyna_params.gamma*getMax(q_value, next_state_) - q_value[tuple(state_)][action_])        
        e.state = next_state

        step += 1
        if step >= max_step:
            break
    return list_reward, avg_list_reward
def main():
    en = env()
    dyna_params = DynaParamas()

    max_step = 10000

    q_value = {}

    model = TrivialModel()

    result, avg_result = dyna_q(q_value, model, en, dyna_params, max_step)

    print(avg_result)
    df=pd.DataFrame({'x': range(max_step), 'y_1': avg_result})
    plt.xlabel("Time Slot")
    plt.ylabel("Time Average Cost")
    plt.plot('x', 'y_1', data=df, marker='o', markevery = int(max_step/10), color='red', linewidth=1, label="planing")
    plt.legend()
    plt.grid()
    plt.show()
if __name__ == '__main__':
    main()

