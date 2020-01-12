import random
from dqn.dqn_db import insert_mem, get_all_memory


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

            insert_mem(conn, transition)

    def load(self, conn):
        temp_memory = get_all_memory(conn)
        for x in range(len(temp_memory)):
            self.memory.append(temp_memory[x])
