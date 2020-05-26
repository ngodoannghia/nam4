import numpy as np

class operator:

    def choose_action(self, q_value, e, dyna_params):
        if np.random.binomial(1, dyna_params.epsilon) == 1:
            a = np.random.uniform(0.0, 1)
            return a
        else:
            if tuple(e.state) not in q_value.keys():
                return np.random.uniform(0.0, 1)
            else:            
                values = q_value[tuple(e.state)]
                max = -1000000
                action = 0
                for tmp in values.keys():
                    if values[tmp] > max:
                        max = values[tmp]
                        action = tmp
            #   print(tuple(e.state), action)
                return action

    def getMax(self, q_value,state):
        max = -100000
        values = q_value[tuple(state)]
        for tmp in values.keys():
            if values[tmp] > max:
                max = values[tmp]       
        return max

    def check(self, q_value, state):
        if tuple(state) not in q_value.keys():
            return True
        else:
            return False
    

