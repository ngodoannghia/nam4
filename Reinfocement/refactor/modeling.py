import numpy as np 
from copy import deepcopy
import heapq

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