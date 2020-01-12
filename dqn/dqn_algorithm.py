from dqn import dqn_constants as cons
from dqn.dqn_nn import DQN
from dqn.dqn_db import get_frequency
import torch.nn as nn
import torch.optim as optim
from utils import load_model, save_model, preprocess_frame
import os.path
import torch
import random
import numpy as np


class QNetAgent(object):
    def __init__(self, memory):

        self.atari_nn = DQN().to(cons.device)
        self.atari_target_nn = DQN().to(cons.device)

        self.loss_func = nn.MSELoss()
        # self.loss_func = nn.SmoothL1Loss()

        self.optimizer = optim.Adam(params=self.atari_nn.parameters(), lr=cons.learning_rate)
        # self.optimizer = optim.RMSprop(params=test_nn.parameters(), lr=learning_rate)

        self.number_of_frames = 0

        self.memory = memory

        if cons.model_load and os.path.exists(cons.model_file):
            print("Loading previous saved model")
            self.atari_nn.load_state_dict(load_model(cons.model_file))

    def select_action(self, state):

        random_for_greedy = torch.rand(1)[0]

        if random_for_greedy > cons.greedy:
            with torch.no_grad():

                state = preprocess_frame(state, cons.device)
                action_from_nn = self.atari_nn(state)
                action = torch.max(action_from_nn, 1)[1]
                action = action.item()
        else:
            action = random.randrange(0, 13)

        return action

    def select_action_egreedy(self, state, epsilon_egreedy):

        random_for_egreedy = torch.rand(1)[0]

        if random_for_egreedy > epsilon_egreedy:
            with torch.no_grad():

                state = preprocess_frame(state, cons.device)
                action_from_nn = self.atari_nn(state)
                action = torch.max(action_from_nn, 1)[1]
                action = action.item()
        else:
            action = random.randrange(0, 13)

        return action

    def select_action_boltzmann(self, state, temperature, epsilon_boltz):

        random_for_egreedy = torch.rand(1)[0]

        if random_for_egreedy > epsilon_boltz:
            with torch.no_grad():

                state = preprocess_frame(state, cons.device)
                action_from_nn = self.atari_nn(state)
                action = torch.max(action_from_nn, 1)[1]
                action = action.item()
        else:

            state = preprocess_frame(state, cons.device)
            action_from_nn = self.atari_nn(state)
            action_from_nn = action_from_nn.cpu()
            action_from_nn = action_from_nn.detach().numpy()
            action_from_nn = action_from_nn[0]

            expected_reward_array = []

            for x in range(14):
                temp_reward, reward_frequency = get_frequency(conn, x)

                expected_reward = action_from_nn[x] + cons.boltzmann_weight * reward_frequency * temp_reward

                expected_reward_array.append(expected_reward)

            exponent = np.true_divide(expected_reward_array - np.max(expected_reward_array), temperature)

            action_probs = np.exp(exponent) / np.sum(np.exp(exponent))

            action = np.random.choice(14, p=action_probs)

        return action

    def optimize(self):

        if len(self.memory) < cons.batch_size:
            return

        state, action, new_state, reward, done = self.memory.sample(cons.batch_size)

        state = [preprocess_frame(frame, cons.device) for frame in state]
        state = torch.cat(state)

        new_state = [preprocess_frame(frame, cons.device) for frame in new_state]
        new_state = torch.cat(new_state)

        action = cons.LongTensor(action).to(cons.device)
        reward = cons.Tensor(reward).to(cons.device)
        done = cons.Tensor(done).to(cons.device)

        new_state_values = self.atari_target_nn(new_state).detach()
        max_new_state_values = torch.max(new_state_values, 1)[0]
        target_value = reward + (1 - done) * cons.gamma * max_new_state_values

        predicted_value = self.atari_nn(state).gather(1, action.unsqueeze(1)).squeeze(1)

        loss = self.loss_func(predicted_value, target_value)

        self.optimizer.zero_grad()

        loss.backward()

        if cons.clip_error:
            for param in self.atari_nn.parameters():
                param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

        if self.number_of_frames % cons.update_target_frequency == 0:
            self.atari_target_nn.load_state_dict(self.atari_nn.state_dict())

        if self.number_of_frames % cons.save_model_frequency == 0:
            save_model(self.atari_nn, cons.model_file)

        self.number_of_frames += 1
