# code from Richard Lentz

# Used Mixed selection. First Random then boltzmann

# import vrep
import sim as vrep
import sys
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os.path
import cv2
import psutil
import sqlite3
from sqlite3 import Error
import io

from collections import deque

# Checks if Cuda is available
use_cuda = torch.cuda.is_available()
# Sets Device for Calculations
device = torch.device("cuda:0" if use_cuda else "cpu")

# Specify Name for Plots, Recordings, & Model
default_name = "Baxter_ver3X-8_test1"
# Sets the name for the Model file
model_file = default_name + "_model.pth"
file_name = default_name + "_results"
# Sets name for PNG Plot
plot_name = default_name + "_results.png"

database = default_name + ".db"

# Sets if Results are written to a csv file
write_to_file = True
# Sets if Error Clipping is Used
clip_error = True
# Sets if Image is Normalized
normalize_image = True
# Sets if Seed is used
set_seed = True
# Sets if a Model should be loaded
model_load = False
# Sets if memory is saved to DB
mem_save = False
# Sets if memory is loaded from DB
mem_load = False

# Sets style for Matplotlib
plt.style.use("ggplot")

solved_distance = 0.12

# Learning rate of Neural Network
learning_rate = 0.0001
# Gamma for Neural Network Learning
gamma = 0.99
# Sets Number of Hidden Layers for Neural Network
hidden_layer = 512
# Sets Number of Stacked Frames for Input to Neural Network
num_frames_stacked = 4
# Output Size for Network - Num of possible actions
output_size = 14

# Sets Number of Training Episodes
num_frames = 300000

# Sets Size of Replay Buffer
replay_mem_size = 75000
# Sets Size of Replay Batch for Learning
batch_size = 32

# Sets if Boltzmann Selection is used
boltzmann = True
# Sets weight for boltzmann selection
boltzmann_weight = 5
# Sets max temperature for boltzmann selection
max_temperature = 1000
# Sets min temperature for boltzmann selection
min_temperature = 1
# Sets temperature decay for boltzmann selection
decay_temperature = .01

# Sets the Frequency to Update Target Network
update_target_frequency = 5000
# Sets the interval for Saving Model by Frames
save_model_frequency = 10000
# Sets the interval for Updating Plots
report_interval = 10
# Sets the interval for video recording
video_interval = 1

# Sets value of Seed
seed_value = 23

greedy = .1

# Sets egreedy Value
egreedy = 0.9
# Sets Min Value for Egreedy
egreedy_final = 0.1
# Sets Egreedy Decay
egreedy_decay = 15000

Tensor = torch.Tensor
LongTensor = torch.LongTensor


class Vrep_Sim(object):
    
    def __init__(self):
        # Close any open connections
        vrep.simxFinish(-1)
        
        # Create Var for client connectin
        self.clientID = vrep.simxStart('127.0.0.1', 19999, True, True, 5000, 5)
                
        if self.clientID != -1:
            print('Connected to remote API server')
            
            self.joint_array = []
            self.joint_org_position = []
            errorCode, self.input_cam = vrep.simxGetObjectHandle(self.clientID, 'input_camera', vrep.simx_opmode_oneshot_wait)
            errorCode, self.hand = vrep.simxGetObjectHandle(self.clientID, 'Baxter_rightArm_camera', vrep.simx_opmode_oneshot_wait)
            errorCode, self.target = vrep.simxGetObjectHandle(self.clientID, 'right_target', vrep.simx_opmode_oneshot_wait)
            errorCode, self.video_cam = vrep.simxGetObjectHandle(self.clientID, 'video_camera', vrep.simx_opmode_oneshot_wait)
            errorCode, self.main_target = vrep.simxGetObjectHandle(self.clientID, 'target', vrep.simx_opmode_oneshot_wait)
            error, self.right_arm_collision_target = vrep.simxGetCollisionHandle(self.clientID, "right_arm_collision_target#", vrep.simx_opmode_blocking)
            error, self.right_arm_collision_table = vrep.simxGetCollisionHandle(self.clientID, "right_arm_collision_table#", vrep.simx_opmode_blocking)

            # Used to translate action to joint array position
            self.joint_switch = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3, 8: 4, 9: 4, 10: 5, 11: 5, 12: 6, 13: 6}
            
            for x in range(1,8):
                errorCode, joint = vrep.simxGetObjectHandle(self.clientID, 'Baxter_rightArm_joint'+str(x), vrep.simx_opmode_oneshot_wait)
                self.joint_array.append(joint)
           
            for x in range(0, 7):
                vrep.simxGetJointPosition(self.clientID, self.joint_array[x], vrep.simx_opmode_streaming)

            for x in range(0, 7):
                errorCode, temp_pos = vrep.simxGetJointPosition(self.clientID, self.joint_array[x], vrep.simx_opmode_buffer)
                self.joint_org_position.append(temp_pos)

            errorCode, self.xyz_hand = vrep.simxGetObjectPosition(self.clientID, self.hand, -1, vrep.simx_opmode_streaming)
            errorCode, self.xyz_target = vrep.simxGetObjectPosition(self.clientID, self.target, -1, vrep.simx_opmode_streaming)
            errorCode, self.xyz_main_target = vrep.simxGetObjectPosition(self.clientID, self.main_target, -1, vrep.simx_opmode_streaming)
            vrep.simxGetVisionSensorImage(self.clientID, self.input_cam, 0, vrep.simx_opmode_streaming)
            vrep.simxGetVisionSensorImage(self.clientID, self.video_cam, 0, vrep.simx_opmode_streaming)
            errorCode, self.right_arm_collision_state_target = vrep.simxReadCollision(self.clientID, self.right_arm_collision_target, vrep.simx_opmode_streaming)
            errorCode, self.right_arm_collision_state_table = vrep.simxReadCollision(self.clientID, self.right_arm_collision_table, vrep.simx_opmode_streaming)

            time.sleep(1)

        else:
            print('Failed connecting to remote API server')
            sys.exit('Could not connect')
    
    def moveJoint(self, action):
        
        if(action == 0 or action % 2 == 0):
            move_interval = 0.03
        else:
            move_interval = -0.03
            
        joint_num = self.joint_switch.get(action, -1)
        
        errorCode, position = vrep.simxGetJointPosition(self.clientID, self.joint_array[joint_num], vrep.simx_opmode_buffer)
        
        errorCode = vrep.simxSetJointTargetPosition(self.clientID, self.joint_array[joint_num], position + move_interval, vrep.simx_opmode_oneshot_wait)
        
        return errorCode
    
    def calcDistance(self):
        
        errorCode, self.xyz_hand = vrep.simxGetObjectPosition(self.clientID, self.hand, -1, vrep.simx_opmode_buffer)
        
        errorCode, self.xyz_target = vrep.simxGetObjectPosition(self.clientID, self.target, -1, vrep.simx_opmode_buffer)

        # need to check if this formula is calculating distance properly
        distance = math.sqrt(pow((self.xyz_hand[0] - self.xyz_target[0]), 2) + pow((self.xyz_hand[1] - self.xyz_target[1]), 2) + pow((self.xyz_hand[2] - self.xyz_target[2]), 2))
        
        return distance
    
    def getCollisionState(self):
        
        errorCode, self.right_arm_collision_state_target = vrep.simxReadCollision(self.clientID, self.right_arm_collision_target, vrep.simx_opmode_buffer)
        errorCode, self.right_arm_collision_state_table = vrep.simxReadCollision(self.clientID, self.right_arm_collision_table, vrep.simx_opmode_buffer)

        if(self.right_arm_collision_state_target or self.right_arm_collision_state_table):
            
            return True
        
        else:
            
            return False

    def getInputImage(self):
        
        errorCode, resolution, image = vrep.simxGetVisionSensorImage(self.clientID, self.input_cam, 0, vrep.simx_opmode_buffer)
        
        image = np.array(image, dtype=np.uint8)

        image.resize([resolution[0],resolution[1],3])
        
        image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
        
        return image

    def getVideoImage(self):
        
        errorCode, resolution, image = vrep.simxGetVisionSensorImage(self.clientID, self.video_cam, 0, vrep.simx_opmode_buffer)
        
        image = np.array(image, dtype=np.uint8)

        image.resize([resolution[0],resolution[1],3])
        
        image = cv2.rotate(image, cv2.ROTATE_180)
        
        return image

    def displayImage(self):
        
        image = self.getInputImage()
        
        plt.imshow(image)

    def resetSim(self):
        
        for x in range(0,7):
            
            vrep.simxSetJointTargetPosition(self.clientID, self.joint_array[x], self.joint_org_position[x], vrep.simx_opmode_oneshot_wait)

        time.sleep(1)


def create_connection(db_file):

    conn = None
    try:
        conn = sqlite3.connect(db_file, detect_types=sqlite3.PARSE_DECLTYPES)
        return conn
    except Error as e:
        print(e)
 
    return conn


def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out, allow_pickle=True)


# Converts np.array to TEXT when inserting
sqlite3.register_adapter(np.ndarray, adapt_array)

# Converts TEXT to np.array when selecting
sqlite3.register_converter("array", convert_array)



def create_mem_table(conn):
    
    mem_table = """ CREATE TABLE IF NOT EXISTS Mem_Replay (id integer primary key, action integer,reward float, state array, new_state array, done blob); """

    try:
        c = conn.cursor()
        c.execute(mem_table)
    except Error as e:
        print(e)



def create_frequency_table(conn):
    
    frequency_table = """ CREATE TABLE IF NOT EXISTS Frequency (action integer,reward float); """

    try:
        c = conn.cursor()
        c.execute(frequency_table)
    except Error as e:
        print(e)



def drop_mem_table(conn):
    
    drop_mem_table = '''DROP TABLE Mem_Replay;'''
    
    try:
        c = conn.cursor()
        c.execute(drop_mem_table)
    except Error as e:
        print(e)



def insertMem(conn, row):

    sql = ''' INSERT INTO Mem_Replay(action,reward,state,new_state,done) VALUES(?,?,?,?,?) '''
    try:
        cur = conn.cursor()
        cur.execute(sql, row)
        cur.close()
    except Error as e:
        print(e)



def insertFreq(conn, row):

    sql = ''' INSERT INTO Frequency(action,reward) VALUES(?,?) '''
    cur = conn.cursor()
    cur.execute(sql, row)
    return cur.lastrowid


def get_All_Memory(conn):
    
    sql = 'SELECT state, action, new_state, reward, done FROM Mem_Replay;'
    
    cur = conn.cursor()
    cur.execute(sql)
    return cur.fetchall()


def get_frequency(conn, action):
    
    cur = conn.cursor()
    cur.execute("SELECT MAX(reward) FROM Frequency WHERE action=?", (action,))
    
    temp_reward = cur.fetchall()
    temp_reward = temp_reward[0][0]

    if(temp_reward == None):
        
        return 0, 0 
        
    else:
        cur = conn.cursor()
        cur.execute("SELECT * FROM Frequency WHERE action=? AND reward=?", (action,temp_reward,))
     
        rewards = cur.fetchall()
        
        reward_frequency = len(rewards)
        
        return temp_reward, reward_frequency


def calculate_temperature(num_frames):
    temp = -decay_temperature * num_frames
    temp = np.exp(temp)
    temp = temp * max_temperature + 1
    return temp


def calculate_epsilon(frames):
    epsilon = egreedy_final + (egreedy - egreedy_final) * math.exp(-1. * frames / egreedy_decay)
    return epsilon


def load_model():
    return torch.load(model_file)


def save_model(model):
    torch.save(model.state_dict(), model_file)


def preprocess_frame(frame):
    frame = torch.from_numpy(frame)
    frame = frame.to(device, dtype=torch.float32)
    frame = frame.unsqueeze(0)
    return frame


def output_Video(episode, video_array):
    
    out = cv2.VideoWriter(default_name + "_episode-" + str(episode) + ".avi",cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
    
    last_frame = video_array[len(video_array) - 1]
    
    for x in range(5):
        video_array.append(last_frame)
 
    for x in range(len(video_array)):
        out.write(video_array[x])
                
    out.release()


def stack_Frames(stacked_frames, frame, is_new_episode):
    
    if is_new_episode:
        
        stacked_frames = deque([np.zeros((84,84), dtype=np.int) for i in range(num_frames_stacked)], maxlen=4)

        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        
        stacked_state = np.stack(stacked_frames, axis=0)
        
    else:
        stacked_frames.append(frame)

        stacked_state = np.stack(stacked_frames, axis=0) 

    return stacked_state, stacked_frames


def plot_results():
    plt.figure(figsize=(12,5))
    plt.title("Rewards Per Episode")
    plt.plot(rewards_total_episode, alpha=0.6, color='red')
    plt.savefig("episode_reward_" + plot_name)
    plt.close()


class ExperienceReplay(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        
    def push(self, state, action, new_state, reward, done):
        transition = (state, action, new_state, reward, done)
        
        if self.position >= len(self.memory):
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition
        
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        return zip(*random.sample(self.memory, batch_size))
        
    def __len__(self):
        return len(self.memory)
    
    def save(self, conn):
        
        for x in range(len(self.memory)):
            
            temp_mem = self.memory[x]
            
            temp_state = temp_mem[0]
            temp_action = temp_mem[1]
            temp_new_sate = temp_mem[2]
            temp_reward = temp_mem[3]
            temp_done = temp_mem[4]
            
            transition = (temp_action, temp_reward, temp_state, temp_new_sate, temp_done)
            
            insertMem(conn, transition)
            
    def load(self, conn):
        
        temp_memory = get_All_Memory(conn)
        
        for x in range(len(temp_memory)):
            
            self.memory.append(temp_memory[x])


class DQN(nn.Module):
    def __init__(self, in_channels=4, num_actions=18):

        super(DQN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=num_frames_stacked, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.linear1 = nn.Linear(7*7*64, hidden_layer)
        self.linear2 = nn.Linear(hidden_layer, output_size)
        
        # self.activation = nn.Tanh()
        self.activation = nn.ReLU()
        
    def forward(self, x):
        
        if normalize_image:
            x = x / 255
        
        output_conv = self.conv1(x)
        output_conv = self.activation(output_conv)
        output_conv = self.conv2(output_conv)
        output_conv = self.activation(output_conv)
        output_conv = self.conv3(output_conv)
        output_conv = self.activation(output_conv)
        
        output_conv = output_conv.view(output_conv.size(0), -1)
        
        output_linear = self.linear1(output_conv)
        output_linear = self.activation(output_linear)
        output_linear = self.linear2(output_linear)
        
        return output_linear


class QNet_Agent(object):
    def __init__(self):
        
        self.atari_nn = DQN().to(device)
        self.atari_target_nn = DQN().to(device)

        self.loss_func = nn.MSELoss()
        # self.loss_func = nn.SmoothL1Loss()

        self.optimizer = optim.Adam(params=self.atari_nn.parameters(), lr=learning_rate)
        # self.optimizer = optim.RMSprop(params=test_nn.parameters(), lr=learning_rate)
        
        self.number_of_frames = 0
        
        if model_load and os.path.exists(model_file):
            print("Loading previous saved model")
            self.atari_nn.load_state_dict(load_model())

    def select_action(self, state):
        
        random_for_greedy = torch.rand(1)[0]
        
        if random_for_greedy > greedy:
            with torch.no_grad():
                
                state = preprocess_frame(state)
                action_from_nn = self.atari_nn(state)
                action = torch.max(action_from_nn,1)[1]
                action = action.item()
        else:
            action = random.randrange(0,13)
        
        return action
    
    def select_action_egreedy(self, state, epislon_egreedy):
        
        random_for_egreedy = torch.rand(1)[0]
        
        if random_for_egreedy > epislon_egreedy:
            with torch.no_grad():
                
                state = preprocess_frame(state)
                action_from_nn = self.atari_nn(state)
                action = torch.max(action_from_nn,1)[1]
                action = action.item()
        else:
            action = random.randrange(0,13)
        
        return action
    
    def select_action_boltzmann(self, state, temperature, epislon_boltz):
        
        random_for_egreedy = torch.rand(1)[0]
        
        if random_for_egreedy > epislon_boltz:
            with torch.no_grad():
                
                state = preprocess_frame(state)
                action_from_nn = self.atari_nn(state)
                action = torch.max(action_from_nn,1)[1]
                action = action.item()
        else:
            
            state = preprocess_frame(state)
            action_from_nn = self.atari_nn(state)
            action_from_nn = action_from_nn.cpu()
            action_from_nn = action_from_nn.detach().numpy()
            action_from_nn = action_from_nn[0]
            
            expected_reward_array = []
            
            for x in range(14):
                
                temp_reward, reward_frequency = get_frequency(conn, x)
                
                expected_reward = action_from_nn[x] + boltzmann_weight * reward_frequency * temp_reward
                
                expected_reward_array.append(expected_reward)
            
            exponent = np.true_divide(expected_reward_array - np.max(expected_reward_array), temperature)
            
            action_probs = np.exp(exponent) / np.sum(np.exp(exponent))
                    
            action = np.random.choice(14, p=action_probs)

        return action
    
    def optimize(self):
        
        if len(memory) < batch_size:
            return
        
        state, action, new_state, reward, done = memory.sample(batch_size)
        
        state = [preprocess_frame(frame) for frame in state]
        state = torch.cat(state)
        
        new_state = [preprocess_frame(frame) for frame in new_state]
        new_state = torch.cat(new_state)
        
        action = LongTensor(action).to(device)
        reward = Tensor(reward).to(device)
        done = Tensor(done).to(device)
         
        new_state_values = self.atari_target_nn(new_state).detach()
        max_new_state_values = torch.max(new_state_values, 1)[0]
        target_value = reward + (1 - done) * gamma * max_new_state_values
        
        predicted_value = self.atari_nn(state).gather(1, action.unsqueeze(1)).squeeze(1)
        
        loss = self.loss_func(predicted_value, target_value)
        
        self.optimizer.zero_grad()
        
        loss.backward()
        
        if clip_error:
            for param in self.atari_nn.parameters():
                param.grad.data.clamp_(-1,1)
        
        self.optimizer.step()
        
        if self.number_of_frames % update_target_frequency == 0:
            self.atari_target_nn.load_state_dict(self.atari_nn.state_dict())
            
        if self.number_of_frames % save_model_frequency == 0:
            save_model(self.atari_nn)

        self.number_of_frames += 1    


memory = ExperienceReplay(replay_mem_size)

if set_seed:
    torch.manual_seed(seed_value)
    random.seed(seed_value)

if write_to_file:
    file_episode = open(file_name + "_episode.csv" , "w") 
    file_episode.write("Episode,Reward,Avg_Reward_Last_" + str(report_interval) +
                       ",Avg_Reward_Last_100,Avg_Reward_All,Frames_total,Time_Elapsed,Solved,"
                       "Epsilon,Distance,Memory_Usage")
    file_frame = open(file_name + "_frame.csv", "w")
    file_frame.write("Frame,Reward,Avg_Reward_Last_1000,Avg_Reward_Last_10000,Avg_Reward_All,"
                     "Time_Elapsed,Solved,Done,Epsilon,Distance,Memory_Usage")

if mem_save:
    # create a database connection
    conn = create_connection(database)
 
    create_frequency_table(conn)

if mem_load:
    memory.load(conn)

q_agent = QNet_Agent()

sim = Vrep_Sim()
    
rewards_total_episode = []
rewards_total_frame = []

frames_total = 0

solved_after_episode = 0
solved_after_frame = 0
solved = False

boltzmann = True

stacked_frames = 0
episode = 0

boltz_frames = 0

temperature = -1

size = (512, 512)

time.sleep(2)

start_time = time.time()

while frames_total < num_frames:
    
    episode += 1
    
    state = sim.getInputImage()
    
    state, stacked_frames = stack_Frames(stacked_frames, state, True)
    
    score = []
    
    video_array = []
    
    distance = sim.calcDistance()
    
    solved = False
    
    index = 0
    
    if episode % video_interval == 0:
        video_record = True
    else:
        video_record = False
    
    video_array.append(sim.getVideoImage())

    while True:
        
        frames_total += 1
        
        epsilon = calculate_epsilon(frames_total)
        
        action = q_agent.select_action_egreedy(state, epsilon)

        sim.moveJoint(action)
        
        new_distance = sim.calcDistance()
        
        new_state = sim.getInputImage()
        
        new_state, stacked_frames = stack_Frames(stacked_frames, new_state, False)

        video_array.append(sim.getVideoImage())

        if new_distance > distance:
            reward = -1
        elif new_distance == distance:
            reward = 0
        else:
            reward = 1
            
        right_arm_collision_state = sim.getCollisionState()

        if new_distance < solved_distance:
            done = True
            solved = True 
            
        elif right_arm_collision_state:
            solved = False
            done = True
            reward = -1
            
        else:
            done = False
        
        if new_distance > distance:
            index += 1
        else:
            index = 0
        
        if index >= 3:
            done = True
                    
        score.append(reward)
        
        memory.push(state, action, new_state, reward, done)
        q_agent.optimize()
        
        state = new_state
        distance = new_distance
        
        if solved:
            print("Solved on Episode:" + str(episode))
            
        system_info = psutil.virtual_memory()

        if write_to_file:
            
            rewards_total_frame.append(reward)
            mean_reward_1000_frame = sum(rewards_total_frame[-1000:])/1000
            mean_reward_10000_frame = sum(rewards_total_frame[-10000:])/10000
            mean_reward_all_frame = round(sum(rewards_total_frame)/len(rewards_total_frame), 2)
            elapsed_time_frame = time.time() - start_time

            if frames_total >= 10000:
                file_frame.write("\n" + str(frames_total) + "," + str(reward) + "," + str(mean_reward_1000_frame) + "," + str(mean_reward_10000_frame) + "," + str(mean_reward_all_frame) + "," + time.strftime("%H:%M:%S", time.gmtime(elapsed_time_frame)) + "," + str(solved) + "," + str(done) + "," + str(round(epsilon, 2)) + "," + str(distance) + "," + str(system_info.used))
            elif frames_total >= 1000:
                file_frame.write("\n" + str(frames_total) + "," + str(reward) + "," + str(mean_reward_1000_frame) + ",null," + str(mean_reward_all_frame) + "," + time.strftime("%H:%M:%S", time.gmtime(elapsed_time_frame)) + "," + str(solved) + "," + str(done) + "," + str(round(epsilon, 2)) + "," + str(distance) + "," + str(system_info.used))
            else:
                file_frame.write("\n" + str(frames_total) + "," + str(reward) + ",null,null," + str(mean_reward_all_frame) + "," + time.strftime("%H:%M:%S", time.gmtime(elapsed_time_frame)) + "," + str(solved) + "," + str(done) + "," + str(round(epsilon, 2)) + "," + str(distance) + "," + str(system_info.used))
            
        if done:
            rewards_total_episode.append(sum(score))
            
            mean_reward_100 = sum(rewards_total_episode[-100:])/100
            mean_reward_interval = sum(rewards_total_episode[-report_interval:])/report_interval
            mean_reward_all = round(sum(rewards_total_episode)/len(rewards_total_episode), 2)
            
            elapsed_time = time.time() - start_time
            
            if video_record:
                
                output_Video(episode, video_array)
                
            if solved:
                
                output_Video(episode, video_array)
            
            if write_to_file:
                
                if episode < report_interval :
                    file_episode.write("\n" + str(episode) + "," + str(sum(score)) + ",null,null," + str(mean_reward_all) + "," + str(frames_total) + "," + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)) + "," + str(solved) + "," + str(round(epsilon, 2)) + "," + str(distance) + "," + str(system_info.used))
                elif episode < 100:
                    file_episode.write("\n" + str(episode) + "," + str(sum(score)) + "," + str(mean_reward_interval) + ",null," + str(mean_reward_all) + "," + str(frames_total) + "," + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)) + "," + str(solved) + "," + str(round(epsilon, 2)) + "," + str(distance) + "," + str(system_info.used))
                else:
                    file_episode.write("\n" + str(episode) + "," + str(sum(score)) + "," + str(mean_reward_interval) + "," + str(mean_reward_100) + "," + str(mean_reward_all) + "," + str(frames_total) + "," + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)) + "," + str(solved) + "," + str(round(epsilon, 2)) + "," + str(distance) + "," + str(system_info.used))

            if episode % report_interval == 0 and episode > 0:
                
                plot_results()
                
                print("\n*** Episode " + str(episode) + " ***")
                print("Avg_Reward [last " + str(report_interval) + "]: " + str(mean_reward_interval) + ", [last 100]: " + str(mean_reward_100) + ", [all]: " + str(mean_reward_all))
                print("Epsilon: " + str(round(epsilon, 2)))
                print("Temperature: " + str(temperature))
                print("Frames Total: " + str(frames_total))
                print("Elapsed Time: ", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
                print("Memory Usage: " + str(system_info.percent) + "%")
              
            sim.resetSim()   
            break
        
    system_info = psutil.virtual_memory()
    
    if system_info.percent > 98:
        break

plot_results()

if mem_save:
    if mem_load:
        drop_mem_table(conn)
    
    create_mem_table(conn)
    memory.save(conn)

if mem_save:
    conn.commit()
    conn.close()
