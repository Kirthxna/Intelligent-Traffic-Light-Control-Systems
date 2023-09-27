import os, sys
import numpy as np
import traci
import time
import numpy as np
import gym
from collections import deque,  namedtuple
from tqdm import tqdm
import random 
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from torch.autograd import Variable
import torch.nn.utils as utils


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, next_action):
        self.buffer.append((state, action, reward, next_state, done, next_action))

    def sample(self, batch_size):
        state, action, reward, next_state, done, next_action = zip(*random.sample(self.buffer, batch_size))
        return torch.stack([torch.tensor(arr) for arr in state], dim=0), \
               torch.tensor(action, dtype=torch.int64).unsqueeze(1), \
               torch.tensor(reward, dtype=torch.float32, device=device).unsqueeze(1), \
               torch.stack([torch.tensor(arr) for arr in next_state], dim=0), \
               torch.tensor(done, dtype=bool).to(torch.int).unsqueeze(1), \
               torch.tensor(next_action, dtype=torch.int64).unsqueeze(1)

    def __len__(self):
        return len(self.buffer)
    
def initialise_weights(layers):
    for i in range(len(layers)):
        if not isinstance(layers[i], nn.ReLU):
            nn.init.zeros_(layers[i].weight)        
            nn.init.zeros_(layers[i].bias)
    return layers

class ActorNetwork(nn.Module):
    def __init__(self, input_shape, num_of_actions):
        super(ActorNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_shape, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, num_of_actions))
        initialise_weights(self.linear_relu_stack) #initialise weights with std 0.01

    def forward(self, x):
        return F.softmax(self.linear_relu_stack(x.float()), dim=-1)
    
class CriticNetwork(nn.Module):
    def __init__(self, input_shape, num_of_actions):
        super(CriticNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_shape, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, num_of_actions))
        initialise_weights(self.linear_relu_stack)

    def forward(self, x):
        return self.linear_relu_stack(x.float()) 
        
class SumoEnvironment():
    def __init__(self):
        self.green_duration=15
        self.yellow_duration=3
        self.sumoBinary = 'sumo'
        self.options = ["--time-to-teleport","-1","--statistic-output","stats.xml", "--eager-insert","True", \
           "--tripinfo-output","intersection_3_info.xml","--summary-output","summary.xml",\
           "--tripinfo-output.write-unfinished","False"]
        self.sumoCmd = [self.sumoBinary,"-c","intersection_3.sumocfg",*self.options]
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

def plot_and_save(data, out_dir, file_name, x_label='Episode', episode=0, smoothing_window=10, c='c'):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)       
    file_path  = os.path.join(out_dir, file_name +'.txt')       
    with open(file_path, 'w') as f:
        for d in data:
            f.write("%s\n" % d)
        
    smoothened_data = []
    for i in range(1,len(data)):      
        x = sum(data[:i][-smoothing_window:])/len(data[:i][-smoothing_window:])
        smoothened_data.append(x)
    data = smoothened_data
        
    x = [i for i in range(len(data))]
    plt.plot(x, data, c)
    plt.xlabel(x_label)
    plt.ylabel(file_name)
    plt.savefig(out_dir + '/'+ file_name+ str(episode)+f'{episode}.png')
        
output_dir = 'tls_model/'
AGGREGATE_STATS_EVERY = 50 

if 'SUMO_HOME' in os.environ:
	tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
	sys.path.append(tools)
else:
	sys.exit("please declare environment variable 'SUMO_HOME'")

seed=0
episodes=500
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
save_dir="actor_critic/goal"
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
output_dir = 'tsc_actor_critic/'
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
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'terminated', 'next_action'))
actor_net = ActorNetwork(17, 4)
critic_net = CriticNetwork(17, 4)
actor_net.train()
critic_net.train()
actor_optimizer = optim.AdamW(actor_net.parameters(), lr=0.001)
critic_optimizer = optim.AdamW(critic_net.parameters(), lr=0.0001)
loss_list = []
state_size=17
observation_space = gym.spaces.Box(0 , 50, shape=(state_size,1))

replay_buffer = ReplayBuffer(capacity=100000)
training_rewards_per_episode = []
epsilon_values = []

env = SumoEnvironment()
for i in range(episodes):
    env.generate_route_file(dist = 'weibull', n_cars_1=n_cars_1,\
                                            n_steps_1 = n_steps_1, n_steps_2 = n_steps_2,n_steps_3 = n_steps_3, episode=i)
    env.start_sumo()
    traci.simulationStep()
    old_action = random.randint(0, action_size-1)
    old_phase = old_action
    env.set_green_phase(old_action, green_duration)
    step = 1   
    state = env.get_state()
    state = torch.tensor(state, dtype=torch.float32, device=device)
    state = state/observation_space.high[0]
    
    actor_prob_values = actor_net(state).squeeze(0)
    action = np.random.choice(np.array([i for i in range(4)]), p=actor_prob_values.data.numpy())
    
    episode_reward = 0.
    terminated = False
    ep_queue_list = []
    old_reward = 0
    for j in range(max_steps):       
        phase = action
        if phase != old_phase:
            env.set_yellow_phase(old_phase)       
        
        env.set_green_phase(phase, green_duration=green_duration)
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
        next_actor_prob_values = actor_net(next_state).squeeze(0)
        next_action = np.random.choice(np.array([i for i in range(4)]), p=next_actor_prob_values.data.numpy())      
        r = reward * reward_scale
        episode_reward += reward
        
        counter += 1
        replay_buffer.push(state.unsqueeze(0), action, reward, next_state.unsqueeze(0), terminated, next_action)
        action = next_action
        state = next_state
        old_phase = phase        
        if len(replay_buffer) >= 128 and j % 1000 == 0:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch, next_action_batch = replay_buffer.sample(batch_size)
            q_values = critic_net(state_batch).squeeze().gather(1, action_batch)
            next_q_values = critic_net(next_state_batch).squeeze().gather(1, next_action_batch)
            target_q_values = reward_batch + gamma * next_q_values*(1-done_batch)
            td_error = (target_q_values.detach() - q_values)
            critic_loss = (td_error.detach()*q_values).mean()
            actor_probs = actor_net(state_batch).squeeze(1)
            log_probs = torch.log(actor_probs.gather(1, action_batch))
            actor_loss = (-log_probs * td_error.detach()).mean()
            
            critic_optimizer.zero_grad()
            critic_loss.backward()
            
            utils.clip_grad_norm_(critic_net.parameters(), max_norm=1.0)
            critic_optimizer.step()

            actor_optimizer.zero_grad()
            actor_loss.backward()
            
            utils.clip_grad_norm_(actor_net.parameters(), max_norm=1.0)
            actor_optimizer.step()

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


    durations = np.asarray(env.get_from_info(file_path='intersection_3_info.xml'), dtype=np.float32)
    average_travel_time = np.sum(durations)/len(durations)
    print('avg Travel time: ',average_travel_time)
    travel_time_list.append(average_travel_time)

    waiting_times = np.asarray(env.get_from_info(file_path='intersection_3_info.xml',\
                                                retrieve='waitingTime'), dtype=np.float32)
    avg_waiting = np.sum(waiting_times)/len(waiting_times)  
    print('avg waiting time: ',avg_waiting)
    waiting_time_list.append(avg_waiting)


    durations = np.asarray(env.get_from_info(), dtype=np.float32)
    avg_duration = np.sum(durations)/len(durations)    
    print('episode :{}/{}, episode_reward {}, avg_queu {}, avg_time:{}'.format(i, episodes, episode_reward, ep_queue, avg_duration))


    traci.close()
                                
end_time = time.time()

plot_and_save(reward_list, model_path, 'reward', episode=i)
plot_and_save(queue_list, model_path, 'queue', episode=i)
plot_and_save(travel_time_list, model_path, 'avg_travel_time', episode=i)
plot_and_save(waiting_time_list, model_path, 'avg_waiting_time', episode=i)
plot_and_save(epsilon_values, model_path, 'epsilon_values', episode=i)

print("Training took %.2f seconds" % (end_time - start_time))
print('average Travel Time:', sum(travel_time_list[-10:])/len(travel_time_list[-10:]))

file_path  = os.path.join(model_path, 'reward' +'.txt')       
with open(file_path, 'w') as f:
    for d in reward_list:
        f.write("%s\n" % d)
        
file_path  = os.path.join(model_path, 'epsilon_values' +'.txt')       
with open(file_path, 'w') as f:
    for d in epsilon_values:
        f.write("%s\n" % d)

actor_net.eval()
critic_net.eval()
testing_rewards_per_episode = []
testing_waiting_time_list = []
testing_travel_time_list = []
testing_queue_list = []
for i in range(10):
    env.generate_route_file(dist = 'weibull', n_cars_1=n_cars_1,n_steps_1 = n_steps_1, n_steps_2 = n_steps_2,n_steps_3 = n_steps_3, episode=i)
    env.start_sumo()
    traci.simulationStep()
    old_action = random.randint(0, action_size-1)
    old_phase = old_action
    env.set_green_phase(old_action, green_duration)
    step = 1   
    state = env.get_state()
    state = torch.tensor(state, dtype=torch.float32, device=device)
    state = state/observation_space.high[0]
    
    actor_prob_values = actor_net(state).squeeze(0)
    action = torch.argmax(actor_prob_values).item()

    episode_reward = 0.
    terminated = False
    ep_queue_list = []
    old_reward = 0
    for j in range(500):
        phase = action
        if phase != old_phase:
            env.set_yellow_phase(old_phase)       
        
        env.set_green_phase(phase, green_duration=green_duration)
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
        next_actor_prob_values = actor_net(state).squeeze(0)
        next_action = torch.argmax(actor_prob_values).item()
        r = reward * reward_scale
        episode_reward += reward
        
        counter += 1
        action = next_action
        state = next_state
        old_phase = phase        
        if terminated:
           break
    
    episode += 1
    ep_queue = sum(ep_queue_list)/len(ep_queue_list)
    testing_queue_list.append(ep_queue)
    testing_rewards_per_episode.append(episode_reward)
    average_queue = sum(queue_list[-AGGREGATE_REWARD_EVERY:])/len(queue_list[-AGGREGATE_REWARD_EVERY:])

    returns.append(episode_reward)
    total_reward += episode_reward

    try:
        durations = np.asarray(env.get_from_info(file_path='intersection_3_info.xml'), dtype=np.float32)
        average_travel_time = np.sum(durations)/len(durations)
        print('avg Travel time: ',average_travel_time)
        testing_travel_time_list.append(average_travel_time)
        waiting_times = np.asarray(env.get_from_info(file_path='intersection_3_info.xml', retrieve='waitingTime'), dtype=np.float32)
        avg_waiting = np.sum(waiting_times)/len(waiting_times)  
        print('avg waiting time: ',avg_waiting)
        testing_waiting_time_list.append(avg_waiting)
    except:
        print('travel time skipped')
    try:
        durations = np.asarray(env.get_from_info(), dtype=np.float32)
        avg_duration = np.sum(durations)/len(durations)    
        print('episode :{}/{}, episode_reward {}, avg_queu {}, avg_time:{}'.format(i, episodes, episode_reward, ep_queue, avg_duration))
    except:
        print('info file error')

    if i % 50 == 0:
        plot_and_save(testing_rewards_per_episode, model_path, 'testing_reward', episode=i)
        plot_and_save(testing_queue_list, model_path, 'testing_queue', episode=i)
        plot_and_save(testing_travel_time_list, model_path, 'testing_avg_travel_time', episode=i)
        plot_and_save(testing_waiting_time_list, model_path, 'testing_avg_waiting_time', episode=i)

    traci.close()
                                
end_time = time.time()
print("Testing took %.2f seconds" % (end_time - start_time))
print('average Travel Time:', sum(travel_time_list[-10:])/len(travel_time_list[-10:]))

file_path  = os.path.join(model_path, 'testing_rewards_per_episode' +'.txt')       
with open(file_path, 'w') as f:
    for d in testing_rewards_per_episode:
        f.write("%s\n" % d)
