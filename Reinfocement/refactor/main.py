from modeling import *
import numpy as np
from agent import *
from operators import operator
from env import env
import matplotlib.pyplot as plt
import pandas as pd
from paramater import DynaParamater
import os
my_path = os.path.abspath('result')
import time 


def main():
    start_time = time.time()


    en = env()
    op = operator()
    max_step = 10000

    q_value = {}  
    model = Priority_model()
    params_dyna = DynaParamater()
    params_dyna.theta = 0.0001
    params_dyna.planning_steps = 10
    params_dyna.alpha = 0.1
    params_dyna.gamma = 0.95

    sweep_priority = sweeping_priority(op)
    backups, avg_list_reward_bak, avg_list_reward_bat, avg_list_reward_delay, avg_list_total_cost  = sweep_priority.priority_sweeping(q_value, model, en, params_dyna, max_step)
    
    #print(avg_list_reward_bat)
    #print(avg_list_reward)

    # d = dyna(op)
    # q_value.clear()
    # model = TrivialModel()
    # params_dyna.planning_steps = 5
    # params_dyna.alpha = 0.1
    # params_dyna.gamma = 0.95
    # avg_list_reward, avg_list_reward_bat_bak, avg_list_reward_bat = d.dyna_q(q_value, model, en, params_dyna, max_step)


    #print(avg_list_reward)

    #df=pd.DataFrame({'x': range(max_step), 'Cost_total':avg_reward})
    df=pd.DataFrame({'x': range(max_step), 'Cost_delay': avg_list_reward_delay, 'Cost_bakup' : avg_list_reward_bak, 'Cost_battery':avg_list_reward_bat, 'Cost_total': avg_list_total_cost})
   
    plt.xlabel("Time Slot")
    plt.ylabel("Time Average Cost")
    plt.plot('x', 'Cost_delay', data=df, marker='o', markevery = int(max_step/10), color='red', linewidth=1, label="cost_delay")
    plt.plot('x', 'Cost_bakup', data=df, marker='x', markevery = int(max_step/10), color='green', linewidth=1, label="cost_bakup_battery")
    plt.plot('x', 'Cost_battery', data=df, marker='*', markevery = int(max_step/10), color='blue', linewidth=1, label="cost_battery")
    plt.plot('x', 'Cost_total', data=df, marker='+', markevery = int(max_step/10), color='gray', linewidth=1, label="cost_total")
    plt.legend()
    plt.grid()
    # my_file = 'planningwithtabular'+'_.xlsx'
    # export_excel = df.to_excel (os.path.join(my_path, my_file), index = None, header=True)
    # my_file = '000_t.png'
    # plt.savefig(os.path.join(my_path, my_file))
    end_time = time.time()
    print('total run-time: %f ms' % ((end_time - start_time) * 1000))
    plt.show()

if __name__ == '__main__':
    main()
        