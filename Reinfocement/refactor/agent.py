from operators import operator
import numpy as np

class dyna:

    def __init__(self, op):
        self.op = op

    def dyna_q(self,q_value, model, e, dyna_params, max_step):

        e.state = e.init_state()
 
        step = 0

        list_reward = []
        list_reward_bak_bat = []
        list_reward_bat = []

        avg_list_reward = []
        avg_list_reward_bak_bat =[]
        avg_list_reward_bat = []


        while True:
        
            action = self.op.choose_action(q_value, e, dyna_params)

            next_state, reward, done, _ = e.step(action)

            list_reward.append(e.reward_time + e.reward_bak + e.reward_bat)
            list_reward_bak_bat.append(e.reward_bak+e.reward_bat)
            list_reward_bat.append(e.reward_bat)
            #list_reward.append(e.reward_bat)
            avg_list_reward.append(np.mean(list_reward[:]))
            avg_list_reward_bak_bat.append(np.mean(list_reward_bak_bat[:]))
            avg_list_reward_bat.append(np.mean(list_reward_bat[:]))

            if self.op.check(q_value, e.state):
                q_value[tuple(e.state)] = {}
                q_value[tuple(e.state)][action] = 0
            if self.op.check(q_value, next_state):
                q_value[tuple(next_state)] = {}
                q_value[tuple(next_state)][action] = 0

            q_value[tuple(e.state)][action] += dyna_params.alpha *(
                        reward + dyna_params.gamma * self.op.getMax(q_value, next_state) - q_value[tuple(e.state)][action])
          
            model.feed(e.state, action, next_state, reward)

            for t in range(0, dyna_params.planning_steps):
                state_, action_, next_state_ , reward_ = model.sample()

                if self.op.check(q_value, state_):
                    q_value[tuple(state_)] = {}
                    q_value[tuple(state_)][action_] = 0

                if self.op.check(q_value, next_state_):
                    q_value[tuple(next_state_)] = {}
                    q_value[tuple(next_state_)][action_] = 0

                q_value[tuple(state_)][action_] += dyna_params.alpha *(
                    reward + dyna_params.gamma * self.op.getMax(q_value, next_state_) - q_value[tuple(state_)][action_])        

            e.state = next_state
            step += 1
            if step >= max_step:
                break
        return avg_list_reward, avg_list_reward_bak_bat, avg_list_reward_bat

class sweeping_priority:
    def __init__(self, op):
        self.op = op
    def priority_sweeping(self,q_value, model, e, dyna_params, max_step):

        e.state = e.init_state()
        list_reward = []
        list_reward_bak = []
        list_reward_bat = []
        list_total_cost = []

        avg_list_reward = []
        avg_list_reward_bak =[]
        avg_list_reward_bat = []
        avg_list_total_cost = []

        steps = 0

        backups = 0
        
        while True:
            
            if steps >= max_step:
                break
            #print(e.state)
           # print("State: ", e.state)
            # if steps % 96:
            #     e.state = e.init_state()
            action = self.op.choose_action(q_value, e, dyna_params)

            # for a in q_value.keys():
            #     for b in q_value[a].keys():
            #         print(q_value[a][b])

            #print("Step: ",steps," action:", action)

            next_state, reward, done, _ = e.step(action)

            #list_reward.append(e.reward_time)
            list_reward_bak.append(e.reward_bak)
            list_reward_bat.append(e.reward_bat)
            list_reward.append(e.reward_time)
            list_total_cost.append(e.reward_time + e.reward_bat + e.reward_bak)
            #list_reward.append(e.reward_bat)
            #avg_list_reward.append(np.mean(list_reward[:]))
            avg_list_reward_bak.append(np.mean(list_reward_bat[:]))
            avg_list_reward_bat.append(np.mean(list_reward_bat[:]))
            avg_list_reward.append(np.mean(list_reward[:]))
            avg_list_total_cost.append(np.mean(list_total_cost[:]))

            model.feed(e.state, action, next_state, reward)

            if self.op.check(q_value, e.state):
                q_value[tuple(e.state)] = {}
                q_value[tuple(e.state)][action] = 0

            if self.op.check(q_value, next_state):
                q_value[tuple(next_state)] = {}
                q_value[tuple(next_state)][action] = 0

            priority = np.abs(reward + dyna_params.gama * self.op.getMax(q_value, next_state)-q_value[tuple(e.state)][action])

            if priority > dyna_params.theta:
                model.insert(priority, e.state, action)
            
            planning_steps = 0

            while planning_steps < dyna_params.planning_steps and not model.empty():
                priority, state_, action_, next_state_, reward_ = model.sample()
                
                if self.op.check(q_value, state_):
                    q_value[tuple(state_)] = {}
                    q_value[tuple(state_)][action_] = 0
                
                if self.op.check(q_value, next_state_):
                    q_value[tuple(next_state_)] = {}
                    q_value[tuple(next_state_)][action_] = 0

                delta = reward_ + dyna_params.gamma * self.op.getMax(q_value, next_state_) - q_value[tuple(state_)][action_]
                
                q_value[tuple(state_)][action_] += dyna_params.alpha * delta
        

                for state_pre, action_pre, reward_pre in model.predecessor(state_):
        
                    if self.op.check(q_value, state_pre):
                        q_value[tuple(state_pre)] = {}
                        q_value[tuple(state_pre)][action_pre] = 0 

                    priority = np.abs(reward_pre + dyna_params.gamma * self.op.getMax(q_value, state_) - \
                                    q_value[tuple(state_pre)][action_pre])

                    if priority > dyna_params.theta:
                        model.insert(priority, state_pre, action_pre)
                planning_steps += 1
            e.state = next_state
            steps += 1
            backups += planning_steps + 1
        return backups, avg_list_reward_bak, avg_list_reward_bat, avg_list_reward, avg_list_total_cost