import numpy as np
import math

class env():
    def __init__(self):
        self.timeslot = 0.25  # hours, ~15min
        self.batery_capacity = 2000  # Wh
        self.server_service_rate = 20  # units/sec

        self.lamda_high = 100  # units/second
        self.lamda_low = 10
        self.b_high = self.batery_capacity / self.timeslot  # W
        self.b_low = 0
        self.h_high = 0.06*5  # s/unit
        self.h_low = 0.02*5
        self.e_low = 0
        self.e_high = 2
        self.back_up_cost_coef = 0.15
        self.normalized_unit_depreciation_cost = 0.01
        self.max_number_of_server = 15
        self.action = int(np.random.uniform(0, 1)*self.b_high)
        self.priority_coefficent = 0.1

        # power model
        self.d_sta = 300
        self.coef_dyn = 0.5
        self.server_power_consumption = 150

        self.time_steps_per_episode = 96
        self.episode = 0
        self.state = [0, 0, 0, 0]
        self.time = 0
        self.time_step = 0

        self.d_op = 0
        self.d_com = 0
        self.d = 0
        self.m = 0
        self.mu = 0
        self.g = 0

        self.reward_time = 0
        self.reward_bak = 0
        self.reward_bat = 0



    def get_lambda(self):
        return np.random.uniform(self.lamda_low,self.lamda_high)
    def get_b(self):
        b = self.state[1]

        if self.d_op > b:
            return b + self.g
        else:
            if self.g >= self.d:
                return np.minimum(self.b_high, b + self.g - self.d)
            else:
                return b + self.g - self.d
    def get_h(self):
        return np.random.uniform(self.h_low, self.h_high)
    def get_e(self):
        if self.time >= 9 and self.time < 15:
            return 2
        if self.time < 6 or self.time >= 18:
            return 0
        return 1
    def init_state(self):
        return [self.get_lambda(), self.get_b(), self.get_h(), self.get_e()]
    def get_time(self):
        self.time += 0.25
        if self.time == 24:
            self.time = 0
    def get_g(self):
        e = self.state[3]
        if e == 0:
            return np.random.exponential(60) + 100
        if e == 1:
            return np.random.normal(520, 130)
        return np.random.normal(800, 95)
    
    #Check Mi(t) : Coong viec duoc load len local server
    def check_Mi_t(self, m, mu):
        if mu > self.state[0] or mu < 0:
            return False
        if isinstance(self.mu, complex):
            return False
        if m*self.server_service_rate <= mu:
            return False
        return True
    def cost_delay_local_function(self, m, mu):
        if m == 0 and mu == 0:
            return 0
        return mu / (m * self.server_service_rate - mu)
    def cost_delay_cloud_function(self, mu, h, lamda):
        return (lamda - mu)*h
    def cost_function(self, m, mu, h, lamda):
        return self.cost_delay_local_function(m, mu) + self.cost_delay_cloud_function(mu, h, lamda)
    def get_m_mu(self, de_action):
        lamd, _, h, _ = self.state
        opt_val = math.inf
        ans = [-1, -1]
        for m in range(1, self.max_number_of_server + 1):
            normalized_min_cov = self.lamda_low
            mu = (de_action - self.server_power_consumption*m) * normalized_min_cov / self.server_power_consumption
            valid = self.check_Mi_t(m, mu)
            if valid:
                if self.cost_function(m, mu, h, lamd) < opt_val:
                    ans = [m, mu]
                    opt_val = self.cost_function(m, mu, h, lamd)
        return ans

    def get_dop(self):
        return self.d_sta + self.coef_dyn* self.state[0]
    
    def get_dcom(self, m, mu):
        normalized_min_cov = self.lamda_low
        return self.server_power_consumption * m + self.server_power_consumption / normalized_min_cov * mu

    def cal(self, action):
        lamda, b, h, _ = self.state
        d_op = self.get_dop()
        if b <= d_op + 150:
            return [0, 0]
        else:
            low_bound = 150
            high_bound = np.minimum(b - d_op, self.get_dcom(self.max_number_of_server,lamda))
            de_action = low_bound + action * (high_bound - low_bound)

            return self.get_m_mu(de_action)
    
    def reward_func(self, action):
        lamda, b, h, _ = self.state
        cost_delay_wireless = 0
        self.m, self.mu = self.cal(action)
        cost_delay = self.cost_function(self.m, self.mu, h, lamda) + cost_delay_wireless
        if self.d_op > b:
            cost_batery = 0
            cost_bak = self.back_up_cost_coef * self.d_op
        else:
            cost_batery = self.normalized_unit_depreciation_cost * np.maximum(self.d - self.g, 0)
            cost_bak = 0
        cost_bak = cost_bak * self.priority_coefficent
        cost_batery = cost_batery * self.priority_coefficent
        cost_delay = cost_delay * (1 - self.priority_coefficent)
        self.reward_bak = cost_bak
        self.reward_bat = cost_batery
        self.reward_time = cost_delay
        cost = cost_delay + cost_batery + cost_bak

        return cost        
    def step(self, action):
        done = False
        action = float(action)
        self.get_time()
        state = self.state
  
        self.time_step += 1
 
        self.g = self.get_g()


        self.d_op = self.get_dop()
        self.m, self.mu = self.cal(action)
        self.d_com = self.get_dcom(self.m, self.mu)
        reward = self.reward_func(action)
        lambda_t = self.get_lambda()
        b_t = self.get_b()
        h_t = self.get_h()
        e_t = self.get_e()
        self.state = np.array([lambda_t, b_t, h_t, e_t])

        if  self.time_step >= self.time_steps_per_episode:
            done = True
            self.episode += 1
        return self.state, 1 / reward, done, {}
    def reset(self):
        self.state = np.array([self.lamda_low, self.b_high, self.h_low, self.e_low])
        self.time = 0
        self.time_step = 0
        return self.state
      
    