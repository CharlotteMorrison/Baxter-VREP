
import td3.constants as cons
from td3.actor import Actor
from td3.critic import Critic
import numpy as np
import torch
import torch.nn.functional as F


class TD3(object):
    """Agent class that handles the training of the networks
    and provides outputs as actions.
    """

    def __init__(self):
        state_dim = cons.STATE_DIM.flatten().shape[0]
        action_dim = cons.ACTION_DIM
        self.actor = Actor(state_dim, action_dim, cons.MAX_ACTION).to(cons.DEVICE)
        self.actor_target = Actor(state_dim,  action_dim, cons.MAX_ACTION).to(cons.DEVICE)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)  # or 1e-3

        self.critic = Critic(state_dim,  action_dim).to(cons.DEVICE)
        self.critic_target = Critic(state_dim,  action_dim).to(cons.DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)  # or 1e-3

        self.total_it = 0
        self.critic_loss_plot = []
        self.actor_loss_plot = []

    def select_action(self, state, noise=0.02):
        """Select an appropriate action from the agent policy
            Args:
                state (array): current state of environment
                noise (float): how much noise to add to actions
            Returns:
                action (list): nn action results
        """

        state = torch.FloatTensor(state).to(cons.DEVICE).unsqueeze(0).unsqueeze(0)
        action = self.actor(state)
        action = (torch.randn_like(action) * noise).clamp(-cons.NOISE_CLIP, cons.NOISE_CLIP)
        return torch.clamp(action, cons.MIN_ACTION, cons.MAX_ACTION).float()

        # for discretized actions
        # movement = self.convert_action(action)
        # policy_noise = (torch.randn_like(movement) * noise).clamp(-cons.NOISE_CLIP, cons.NOISE_CLIP)
        # movement = torch.add(movement, policy_noise)
        # return action, torch.clamp(movement, cons.MIN_ACTION, cons.MAX_ACTION).float()

    def convert_action(self, action):
        """ reshape torch.Size([1, 21]) to torch.Size([3, 7)] and map values to movements
            :param action: torch
            :return: torch
        """
        action = torch.reshape(action, (3, 7))
        _, index = action.max(0)
        index = torch.sub(index, 1) * .1
        return index

    def train(self, replay_buffer, iterations):
        """Train and update actor and critic networks
            Args:
                replay_buffer (ReplayBuffer): buffer for experience replay
                iterations (int): how many times to run training
            Return:
                actor_loss (float): loss from actor network
                critic_loss (float): loss from critic network
        """
        for it in range(iterations):
            # Sample replay buffer (priority replay)
            state, action, reward, next_state, done, _, _ = replay_buffer.sample(cons.BATCH_SIZE, beta=0.5)

            state = torch.from_numpy(state).float().to(cons.DEVICE)                 # torch.Size([100, 84, 84])
            next_state = torch.from_numpy(next_state).float().to(cons.DEVICE)       # torch.Size([100, 84, 84])
            action = torch.from_numpy(action).float().to(cons.DEVICE)               # torch.Size([100, 7])
            reward = torch.as_tensor(reward, dtype=torch.float32).to(cons.DEVICE)   # torch.Size([100])
            done = torch.as_tensor(done, dtype=torch.float32).to(cons.DEVICE)       # torch.Size([100])

            with torch.no_grad():
                # select an action according to the policy and add clipped noise
                # need to select set of actions

                # next_action = torch.reshape(torch.clamp((self.actor_target(next_state.unsqueeze(1))
                #                             + noise).flatten(), -1, 1), (100, 7))

                next_action = self.actor_target(next_state.unsqueeze(1))
                noise = (torch.rand_like(next_action) *
                         cons.POLICY_NOISE).clamp(-cons.NOISE_CLIP, cons.NOISE_CLIP)
                next_action = torch.reshape(torch.clamp((next_action + noise).flatten(), -1, 1), (100, 7))

                # Compute the target Q value
                target_q1, target_q2 = self.critic(state.float(), next_action.float())
                target_q = torch.min(target_q1, target_q2)

                target_q = reward + (done * cons.GAMMA * target_q).detach()

            # get current Q estimates
            current_q1, current_q2 = self.critic(state.float(), action.float())

            # compute critic loss
            critic_loss = F.mse_loss(current_q1,
                                     target_q[:1, :].transpose(0, 1)) + F.mse_loss(current_q2,
                                                                                   target_q[:1, :].transpose(0, 1))
            # optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            # print('Critic loss: {}'.format(critic_loss.item()))

            self.critic_loss_plot.append(critic_loss.item())
            # delayed policy updates
            if self.total_it % cons.POLICY_FREQ == 0:  # update the actor policy less frequently

                # compute the actor loss
                q_action = self.actor(state.unsqueeze(1)).float().detach()

                # actor_loss = -self.critic.get_q(state, self.actor(state.unsqueeze(1)).float()).mean()
                actor_loss = -self.critic.get_q(state, q_action).mean()

                # optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                # print('Actor loss: {}'.format(actor_loss.item()))
                self.actor_loss_plot.append(actor_loss.item())

                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(cons.TAU * param.data + (1 - cons.TAU) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(cons.TAU * param.data + (1 - cons.TAU) * target_param.data)

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

    def load(self, filename="best_avg", directory="./saves"):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
