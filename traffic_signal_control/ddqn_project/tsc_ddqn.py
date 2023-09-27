import os
import random
import sys
import time
from collections import deque, namedtuple
from datetime import datetime

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import traci
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from collections import deque


class Network(nn.Module):
    def __init__(self, input_shape, num_of_actions):
        super(Network, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_of_actions))

    def forward(self, x):
        x = self.linear_relu_stack(x)
        return x   
        
class SumoEnvironment():
    def __init__(self):
        self.green_duration=15
        self.yellow_duration=3
        self.sumoBinary = 'sumo'
        self.options = ["--time-to-teleport","-1","--statistic-output","stats.xml", "--eager-insert","True", \
           "--tripinfo-output","intersection_3_info.xml","--summary-output","summary.xml",\
           "--tripinfo-output.write-unfinished","False"]
        self.sumoCmd = [self.sumoBinary, "-c","intersection_3.sumocfg", *self.options]
        self.intersection = "intersection"
        
    def start_sumo(self):
        traci.start(self.sumoCmd)
                
    def get_state(self):
        lanes_list = list(traci.trafficlight.getControlledLanes(self.intersection))
        N = len(lanes_list) 
        state = np.zeros((N+1 ,), dtype=np.int32)
        for i,lane in enumerate(lanes_list):           
                lane_state = traci.lane.getLastStepHaltingNumber(lanes_list[i])                                                                           #halting vehicles on lane
                state[i] = lane_state 
        all_phases = traci.trafficlight.getAllProgramLogics(self.intersection)                                                                      #traffic light control id
        current_phase = traci.trafficlight.getRedYellowGreenState(self.intersection)
        for i,phase in enumerate(all_phases[0].getPhases()):
            s = phase.__repr__().split(',')
            s = s[1].split('\'')[1]
            if current_phase == s:
                state[N] = np.floor(i/2)
        return state
    
    def get_queue(self):
        lanes_list = list(traci.trafficlight.getControlledLanes(self.intersection))
        queue_list = []
        for lane in lanes_list:
            q = traci.lane.getLastStepHaltingNumber(lane)
            queue_list.append(q)       
        queue = np.sum(np.asarray(queue_list))        
        return queue    
        
    def set_green_phase(self, action, green_duration=15):
        green_phase_code = action * 2
        for t in range(green_duration):           
            traci.trafficlight.setPhase(self.intersection, green_phase_code)
            traci.simulationStep()    
    
    def set_yellow_phase(self, action, yellow_duration=3):
        yellow_phase_code = action * 2 + 1
        for t in range(yellow_duration):            
            traci.trafficlight.setPhase(self.intersection, yellow_phase_code)
            traci.simulationStep()

    def get_from_info(self, file_path='intersection_3_info.xml', retrieve='duration'):
        retrieve = ' '+ retrieve +'='
        duration_list = []
        with open(file_path,'r') as f:
            text_lines = f.readlines()
            for w in text_lines:
                a = w.split('"')
                for i,value in enumerate(a):
                    if value==retrieve:                       
                        duration_list.append(a[i+1])        
        return duration_list
    
    def generate_route_file(self, dist = 'weibull', n_steps_1=3600, n_steps_2=3600, n_steps_3=3600,n_cars_1=1500,
                        mu=2000, sigma=1000, episode=1, o_file="intersection_3.rou.xml"):
        r = np.random.RandomState(episode)
        if dist=='weibull':
            s_1 = r.weibull(2, n_cars_1)
            mins_1 = min(s_1)
            maxs_1 = max(s_1)
            new_s_1 = []
            for v in s_1:
                v = v*n_steps_1/(maxs_1-mins_1)
                new_s_1.append(v)
            s_1=new_s_1
        
        else:
            s_1 = r.normal(mu, sigma, n_cars_1)
        
        s_1 = np.abs(np.rint(s_1))
        s_1 = sorted(s_1)
        
        max_s_1 = max(s_1)
        s = s_1
        with open(o_file, "w") as routes:
            print('<routes>', file=routes)
            for car_id, t in enumerate(s):
                if r.uniform()<0.75:
                    route_id = r.randint(1, 5)  
                    if route_id == 1:
                        print(f'      <vehicle id="{car_id}" depart="{t}"> \n        <route edges="e1_in e3_out"/> \n      </vehicle>',file=routes)
                    elif route_id ==2 :
                        print(f'      <vehicle id="{car_id}" depart="{t}"> \n        <route edges="e2_in e4_out"/> \n      </vehicle>',file=routes)
                    elif route_id ==3:
                        print(f'      <vehicle id="{car_id}" depart="{t}"> \n        <route edges="e3_in e1_out"/> \n      </vehicle>',file=routes)
                    elif route_id ==4:
                        print(f'      <vehicle id="{car_id}" depart="{t}"> \n        <route edges="e4_in e2_out"/> \n      </vehicle>',file=routes)
                else:
                    if r.uniform()<0.5:
                        
                        route_id = r.randint(1, 5)  
                        if route_id == 1:
                            print(f'      <vehicle id="{car_id}" depart="{t}"> \n        <route edges="e1_in e4_out"/> \n      </vehicle>',file=routes)
                        elif route_id ==2 :
                            print(f'      <vehicle id="{car_id}" depart="{t}"> \n        <route edges="e2_in e1_out"/> \n      </vehicle>',file=routes)
                        elif route_id ==3:
                            print(f'      <vehicle id="{car_id}" depart="{t}"> \n        <route edges="e3_in e2_out"/> \n      </vehicle>',file=routes)
                        elif route_id ==4:
                            print(f'      <vehicle id="{car_id}" depart="{t}"> \n        <route edges="e4_in e3_out"/> \n      </vehicle>',file=routes)
                    else:
                        route_id = r.randint(1, 5)  
                        if route_id == 1:
                            print(f'      <vehicle id="{car_id}" depart="{t}"> \n        <route edges="e1_in e2_out"/> \n      </vehicle>',file=routes)
                        elif route_id ==2 :
                            print(f'      <vehicle id="{car_id}" depart="{t}"> \n        <route edges="e2_in e3_out"/> \n      </vehicle>',file=routes)
                        elif route_id ==3:
                            print(f'      <vehicle id="{car_id}" depart="{t}"> \n        <route edges="e3_in e4_out"/> \n      </vehicle>',file=routes)
                        elif route_id ==4:
                            print(f'      <vehicle id="{car_id}" depart="{t}"> \n        <route edges="e4_in e1_out"/> \n      </vehicle>',file=routes)
            print('</routes>', file=routes)

def plot_and_save(data, out_dir, file_name, save=True,save_data=True, font_size=12, font_weight='bold', x_label='Episode', y_label='y',  style='seaborn-whitegrid',\
                  episode=0, smoothing=True, smoothing_window=10, fig_size=(14,10), c='c', plot=True):
    if save_data:
        save_score_file(data, out_dir=out_dir, file_name=file_name)
        
    plt.style.use(style)
    if smoothing:
        data = smoothing_curve(data,smoothing_window)
    
    font = {'family' : 'normal',
        'weight' : font_weight,
        'size'   : font_size}

    plt.rc('font', **font)
#     plt.rcParams.update({'font.size':14, })
    x = [i for i in range(len(data))]
    fig1 = plt.figure(figsize=fig_size)
    plt.plot(x, data, c)
    plt.xlabel(x_label)
    plt.ylabel(file_name)
    if save:
        plt.savefig(out_dir + '/'+ file_name+ str(episode)+f'{episode}.png')
        
def smoothing_curve(array, window_length=10):
    smoothened = []
    for i in range(1,len(array)):
        
        x = sum(array[:i][-window_length:])/len(array[:i][-window_length:])
        smoothened.append(x)
    return smoothened
    
def save_score_file(data, out_dir='scores_data', file_name='scores'):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    file_path  = os.path.join(out_dir, file_name +'.txt')
    
    with open(file_path, 'w') as f:
        for d in data:
            f.write(str(d)+'\n')

def get_score_file(file_path):
    out_list = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.replace('\n','')
            n = int(float(line))
            out_list.append(n)
  
    return out_list  

output_dir = 'tls_model/'
AGGREGATE_STATS_EVERY = 50 

if 'SUMO_HOME' in os.environ:
	tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
	sys.path.append(tools)
else:
	sys.exit("please declare environment variable 'SUMO_HOME'")

seed=0
episodes=500
evaluation_episodes=5000
batch_size=64
gamma=0.95
initial_memory_threshold=128             
replay_memory_size=20000
epsilon_steps=300
epsilon = 1
epsilon_final=0.01
episode = 0
epsilon_initial = 1.0
reward_scale=1./20
clip_grad=1.
split=False
action_input_layer=0
save_dir="ddqn/goal"
render_freq=100
save_frames=False
visualise=False
title="PDQN"

np.random.seed(seed)
reward_list = []
queue_list  = []
travel_time_list = []
waiting_time_list = []
AGGREGATE_REWARD_EVERY = 50
max_steps = 3800
steps = 3800
final_time =3800
save_label = 'results'
output_dir = 'ddqn/'
model_path = os.path.join(output_dir, save_label)
if not os.path.exists(model_path):
    os.makedirs(model_path)
yellow_duration = 3
green_duration = 15
n_cars_1 = 4001
n_steps_1 = 3600
n_steps_2 = 3600
n_steps_3 = 3600
action_size= 4
total_reward = 0.
returns = []
start_time = time.time()
counter = 0
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'terminated'))
policy_net = Network(17, 4).to(device)
target_net = Network(17, 4).to(device)
target_net.load_state_dict(policy_net.state_dict())
policy_net.train()
target_net.eval()
loss_list = []
optimizer = optim.AdamW(policy_net.parameters())
state_size=17
observation_space = gym.spaces.Box(0 , 50, shape=(state_size,1))
replay_memory = deque([], maxlen=100000)
training_rewards_per_episode = []
epsilon_values = []

env = SumoEnvironment()
for i in range(episodes):
    env.generate_route_file(dist = 'weibull', n_cars_1=n_cars_1,\
                                            n_steps_1 = n_steps_1, n_steps_2 = n_steps_2,n_steps_3 = n_steps_3, episode=i)
    env.start_sumo()
    traci.simulationStep()
    action = random.randint(0, action_size-1)
    env.set_green_phase(action, green_duration)
    step = 1   
    state = env.get_state()
    state = torch.tensor(state, dtype=torch.float32, device=device)
    state = state/observation_space.high[0]
    
    if random.random() < epsilon:
        next_action = random.randint(0, 3)
    else:
        q_values = policy_net(state)
        print(' q_values ', q_values.shape)
        next_action = torch.argmax(q_values).item()

    episode_reward = 0.
    terminated = False
    ep_queue_list = []
    old_reward = 0
    for j in range(max_steps):       
        #phase = action
        if next_action != action:
            env.set_yellow_phase(action)       
        
        env.set_green_phase(next_action, green_duration=green_duration)
        step = step + yellow_duration + green_duration
        next_state = env.get_state()
        new_reward = np.sum(next_state[0:-1]) 
        next_state = torch.tensor(next_state, dtype=torch.float32, device=device)
        next_state = next_state/observation_space.high[0]              
        queue = env.get_queue()
        reward = - new_reward
        ep_queue_list.append(queue)
        next_step = step + yellow_duration + green_duration
        
        r = reward * reward_scale
        replay_memory.append(Transition(state.unsqueeze(0), action, next_state.unsqueeze(0), r, terminated))

        action = next_action

        if next_step >= max_steps:
            terminated = True            
        if random.random() < epsilon:
            next_action = random.randint(0, 3)
        else:
            q_values = policy_net(next_state)
            print(' q_values ', q_values.shape)
            next_action = torch.argmax(q_values).item()
        
        episode_reward += reward
        
        counter += 1  
        state = next_state
               
        if len(replay_memory) >= 128 and j % 10 == 0:
          transitions = random.sample(replay_memory, 128)

          batch = Transition(*zip(*transitions))

          state_batch = torch.cat([torch.tensor(arr) for arr in batch.state], dim=0)
          action_batch = torch.tensor(batch.action).unsqueeze(1)
          reward_batch = torch.tensor(batch.reward).unsqueeze(1)
          next_state_batch = torch.cat([torch.tensor(arr) for arr in batch.next_state], dim=0)
          terminated_batch = torch.tensor(batch.terminated, dtype=bool)
          terminated_batch = terminated_batch.to(torch.int).unsqueeze(1)
          expected_state_action_values = torch.zeros(128, dtype=torch.float32, device=device)
          k = 0
          with torch.no_grad():
              while (k < len(state_batch)):
                  if terminated_batch[k] == 0:
                      expected_policy_net_argmax = torch.argmax(policy_net(next_state_batch[k])).item()
                      expected_target_net_value = target_net(next_state_batch[k])
                      expected_state_action_values[k] = expected_target_net_value[expected_policy_net_argmax]
                  k = k + 1
          expected_state_action_values = expected_state_action_values.unsqueeze(1)*gamma + reward_batch
          next_state_action_values = policy_net(state_batch).gather(1, action_batch)

          criterion = nn.SmoothL1Loss()

          loss = criterion(next_state_action_values, expected_state_action_values)
          loss_list.append(loss)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
        if terminated:
           break
    
    episode += 1
    epsilon_values.append(epsilon)
    epsilon = epsilon*0.98627
    ep_queue = sum(ep_queue_list)/len(ep_queue_list)
    queue_list.append(ep_queue)
    reward_list.append(episode_reward)
    average_queue = sum(queue_list[-AGGREGATE_REWARD_EVERY:])/len(queue_list[-AGGREGATE_REWARD_EVERY:])

    returns.append(episode_reward)
    total_reward += episode_reward
    if i % 100 == 0:
        target_net.load_state_dict(policy_net.state_dict())

    try:
        durations = np.asarray(env.get_from_info(file_path='intersection_3_info.xml'), dtype=np.float32)
        average_travel_time = np.sum(durations)/len(durations)
        print('avg Travel time: ',average_travel_time)
        travel_time_list.append(average_travel_time)
        waiting_times = np.asarray(env.get_from_info(file_path='intersection_3_info.xml', retrieve='waitingTime'), dtype=np.float32)
        avg_waiting = np.sum(waiting_times)/len(waiting_times)  
        print('avg waiting time: ',avg_waiting)
        waiting_time_list.append(avg_waiting)
    except:
        print('travel time skipped')
    try:
        durations = np.asarray(env.get_from_info(), dtype=np.float32)
        avg_duration = np.sum(durations)/len(durations)    
        print('episode :{}/{}, episode_reward {}, avg_queu {}, avg_time:{}'.format(i, episodes, episode_reward, ep_queue, avg_duration))
    except:
        print('info file error')

    traci.close()
                                
end_time = time.time()
print("Training took %.2f seconds" % (end_time - start_time))
print('average Travel Time:', sum(travel_time_list[-10:])/len(travel_time_list[-10:]))

plot_and_save(reward_list, model_path, 'reward', episode=i)
plot_and_save(queue_list, model_path, 'queue', episode=i)
plot_and_save(travel_time_list, model_path, 'avg_travel_time', episode=i)
plot_and_save(waiting_time_list, model_path, 'avg_waiting_time', episode=i)
plot_and_save(epsilon_values, model_path, 'epsilon_values', episode=i)

policy_net.eval()
testing_rewards_per_episode = []
for i in range(10):
    env.generate_route_file(dist = 'weibull', n_cars_1=n_cars_1,\
                                            n_steps_1 = n_steps_1, n_steps_2 = n_steps_2,n_steps_3 = n_steps_3, episode=i)
    env.start_sumo()
    traci.simulationStep()
    action = random.randint(0, action_size-1)
    env.set_green_phase(old_action, green_duration)
    step = 1   
    state = env.get_state()
    state = torch.tensor(state, dtype=torch.float32, device=device)
    state = state/observation_space.high[0]
    
    q_values = policy_net(state)
    print(' q_values ', q_values.shape)
    next_action = torch.argmax(q_values).item()

    episode_reward = 0.
    terminated = False
    ep_queue_list = []
    old_reward = 0
    for j in range(500):
        if next_action != action:
            env.set_yellow_phase(action)       
        
        env.set_green_phase(next_action)
        step = step + yellow_duration + green_duration
        next_state = env.get_state()
        new_reward = np.sum(next_state[0:-1]) 
        next_state = torch.tensor(next_state, dtype=torch.float32, device=device)
        next_state = next_state/observation_space.high[0]              
        queue = env.get_queue()
        reward = - new_reward
        ep_queue_list.append(queue)
        next_step = step + yellow_duration + green_duration
        if next_step >= max_steps:
            terminated = True            
        q_values = policy_net(next_state)
        print(' q_values ', q_values.shape)

        r = reward * reward_scale
        replay_memory.append(Transition(state.unsqueeze(0), action, next_state.unsqueeze(0), r, terminated))

        action = next_action
        next_action = torch.argmax(q_values).item()
        
        episode_reward += reward
        
        counter += 1  
        state = next_state

        if terminated:
           break
    
    episode += 1
    ep_queue = sum(ep_queue_list)/len(ep_queue_list)
    queue_list.append(ep_queue)
    reward_list.append(episode_reward)
    average_queue = sum(queue_list[-AGGREGATE_REWARD_EVERY:])/len(queue_list[-AGGREGATE_REWARD_EVERY:])

    returns.append(episode_reward)
    total_reward += episode_reward
    if i % 100 == 0:
        target_net.load_state_dict(policy_net.state_dict())

    durations = np.asarray(env.get_from_info(file_path='intersection_3_info.xml'), dtype=np.float32)
    average_travel_time = np.sum(durations)/len(durations)
    print('avg Travel time: ',average_travel_time)
    travel_time_list.append(average_travel_time)
    waiting_times = np.asarray(env.get_from_info(file_path='intersection_3_info.xml', retrieve='waitingTime'), dtype=np.float32)
    avg_waiting = np.sum(waiting_times)/len(waiting_times)  
    print('avg waiting time: ',avg_waiting)
    waiting_time_list.append(avg_waiting)

    durations = np.asarray(env.get_from_info(), dtype=np.float32)
    avg_duration = np.sum(durations)/len(durations)    


    traci.close()
                                
end_time = time.time()
plot_and_save(reward_list, model_path, 'testing_reward', episode=i)
plot_and_save(queue_list, model_path, 'testing_queue', episode=i)
plot_and_save(travel_time_list, model_path, 'testing_avg_travel_time', episode=i)
plot_and_save(waiting_time_list, model_path, 'testing_avg_waiting_time', episode=i)

