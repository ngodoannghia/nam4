

class operator:
        
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