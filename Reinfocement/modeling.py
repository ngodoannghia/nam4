import numpy as np 
from copy import deepcopy
from env import env
import heapq
import matplotlib.pyplot as plt
import pandas as pd

class PriorityQueue:
    def __init__(self):
        self.pq = []
        self.entry_finder = {}
        self.REMOVED = '<removed-task>'
        self.counter = 0

    def add_item(self, item, priority = 0):
        if item in self.entry_finder:
            self.remove_item(item)
        entry = [priority, self.counter, item]
        self.counter += 1
        self.entry_finder[item] = entry
        heapq.heappush(self.pq, entry)
    def remove_item(self, item):
        entry = self.entry_finder.pop(item)
        entry[-1] = self.REMOVED
    def pop_item(self):
        while self.pq:
            priority, count, item = heapq.heappop(self.pq)
            if item is not self.REMOVED:
                del self.entry_finder[item]
                return item, priority
        raise KeyError('pop from an empty priority queue')

    def empty(self):
        return not self.entry_finder   



class DynaParamater:
    def __init__(self):
        self.gama = 0.95

        self.epsilon = 0.1

        self.alpha = 0.1

        self.time_weight = 0

        self.planning_steps = 5

        self.runs = 10

        self.theta = 0


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

class Priority_model(TrivialModel):
    def __init__(self, rand = np.random):
        TrivialModel.__init__(self, rand)

        self.priority_queue = PriorityQueue()

        self.predecessors = {}
    
    def insert(self, priority, state, action):
        self.priority_queue.add_item((tuple(state),action), -priority)
    
    def empty(self):
        return self.priority_queue.empty()

    def sample(self):
        (state,action), priority = self.priority_queue.pop_item()
        next_state, reward = self.model[state][action]
        state = deepcopy(state)
        next_state = deepcopy(next_state)

        return -priority, list(state), action, list(next_state), reward
    
    def feed(self, state, action, next_state, reward):
        state = deepcopy(state)
        next_state = deepcopy(next_state)
        TrivialModel.feed(self, state, action, next_state, reward)
        if tuple(next_state) not in self.predecessors.keys():
            self.predecessors[tuple(next_state)] = set()
        self.predecessors[tuple(next_state)].add((tuple(state), action))
    
    def predecessor(self, state):

        if tuple(state) not in self.predecessors.keys():
            return []
        predecessors = []

        for item in self.predecessors[tuple(state)]:
            state_pre, action_pre = list(item)
            predecessors.append([list(state_pre), action_pre, self.model[state_pre][action_pre][1]])
        return predecessors

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

def check(q_value, state):
    if tuple(state) not in q_value.keys():
        return True
    else:
        return False

def priority_sweeping(q_value, model, e, dyna_params, max_step):

    e.state = e.init_state()
    list_reward = []
    avg_list_reward = []
    steps = 1

    backups = 0
    
    while True:
        
        if steps > max_step:
            break
        action = choose_action(q_value, e, dyna_params)

        next_state, reward, done, _ = e.step(action)

        list_reward.append(e.reward_time + e.reward_bak + e.reward_bat)
        avg_list_reward.append(np.mean(list_reward[:]))

        model.feed(e.state, action, next_state, reward)

        if check(q_value, e.state):
            q_value[tuple(e.state)] = {}
            q_value[tuple(e.state)][action] = 0

        if check(q_value, next_state):
            q_value[tuple(next_state)] = {}
            q_value[tuple(next_state)][action] = 0

        priority = np.abs(reward + dyna_params.gama * getMax(q_value, next_state)-q_value[tuple(e.state)][action])

        if priority > dyna_params.theta:
            model.insert(priority, e.state, action)
        
        planning_steps = 0

        while planning_steps < dyna_params.planning_steps and not model.empty():
            priority, state_, action_, next_state_, reward_ = model.sample()
            
            if check(q_value, state_):
                q_value[tuple(state_)] = {}
                q_value[tuple(state_)][action_] = 0
            
            if check(q_value, next_state_):
                q_value[tuple(next_state_)] = {}
                q_value[tuple(next_state_)][action_] = 0

            delta = reward_ + dyna_params.gamma * getMax(q_value, next_state_) - q_value[tuple(state_)][action_]
            
            q_value[tuple(state_)][action_] += dyna_params.alpha * delta

        
            #print(list(model.predecessors[tuple(state_)]))
            #print(state_)
        # a, b , c = model.predecessors(state_)
    

            for state_pre, action_pre, reward_pre in model.predecessor(state_):
    
                if check(q_value, state_pre):
                    q_value[tuple(state_pre)] = {}
                    q_value[tuple(state_pre)][action_pre] = 0 

                priority = np.abs(reward_pre + dyna_params.gamma * getMax(q_value, state_) - \
                                q_value[tuple(state_pre)][action_pre])

                if priority > dyna_params.theta:
                    model.insert(priority, state_pre, action_pre)
            planning_steps += 1
        e.state = next_state
        steps += 1
        backups += planning_steps + 1
    return backups, avg_list_reward

def main():
    q_value = {}
    max_step = 1000
    model = Priority_model()
    params_dyna = DynaParamater()
    params_dyna.planning_steps = 5
    params_dyna.alpha = 0.5
    params_dyna.gamma = 0.95

    en = env()

    backups, avg_list_reward = priority_sweeping(q_value, model, en, params_dyna, max_step)
    
    
  #  print(backups)
  #  print(avg_list_reward)
    df=pd.DataFrame({'x': range(max_step), 'y_1': avg_list_reward})
    plt.xlabel("Time Slot")
    plt.ylabel("Time Average Cost")
    plt.plot('x', 'y_1', data=df, marker='o', markevery = int(max_step/10), color='red', linewidth=1, label="planing")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == '__main__':
    main()
