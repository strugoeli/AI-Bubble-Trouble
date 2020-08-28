import random
import torch
from memory import PrioritizedReplayBuffer
from itertools import count
from qvalues import QValues
import project_utils
from parameters import *
import matplotlib

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython: from IPython import display


class Agent:
    """
    This class ris responsible for training and testing the model
    """

    def __init__(self, policy_net, target_net, strategy, em, test_em, num_actions, optimizer):
        self.curr_step = 0
        self.rate = 0
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net, self.target_net = policy_net, target_net
        self.memory = PrioritizedReplayBuffer(MEMORY_SIZE)
        self.optimizer = optimizer
        self.em = em
        self.test_em = test_em
        self.scale = 0.7

    def select_action(self, state, actions):
        """
        This method is given state and actions that can be taken from this state
        then sample randomly number k in (0,1].
        if rate > k samples randomly number from the given actions  and returns it
        else returns the action with the highest q-value from the given actions

        :param state: Tensor
        :param actions: list of int - action space for the given state
        :return: int - selected action
        """
        self.rate = self.strategy.get_exploration_rate(self.curr_step)
        self.curr_step += 1
        if self.rate > random.random():
            return random.choice(actions)
        else:
            with torch.no_grad():
                return self.policy_step(state, actions)

    def policy_step(self, state, actions):
        """
        This method is given state and actions that can be taken from this state and returns
        the action with the highest q-value from the given actions
        :param state: Tensor
        :param actions: list of int - action space for the given state
        :return: int - selected action
        """
        with torch.no_grad():
            output = self.policy_net(state.to(self.device))
            top = output.topk(4).indices[0].cpu().numpy()
            for act in top:
                if act in actions:
                    return act

    def test(self, n_episodes):
        """
        This method is given the number of episides to test the model and tests it
        :param n_episodes: number of episodes to train the model
        :return: The average reward of the test
        """
        avg_reward = 0
        for episode in range(n_episodes):
            state = self.test_em.reset()
            total_reward = 0.0
            for _ in count():

                action = self.policy_step(state, self.test_em.get_legal_actions())
                next_state, reward, done, info = self.test_em.step(action)
                total_reward += reward
                state = next_state

                if done:
                    print("Finished Episode {} with reward {}".format(episode, total_reward))
                    avg_reward += total_reward
                    break

        avg_reward = avg_reward / n_episodes if n_episodes else 0
        print("Avarage reward over {} espisodes is: {}".format(n_episodes, avg_reward))
        return avg_reward / n_episodes

    def train(self, num_episodes):
        """
        This method is given the number of episides to train the model and trains it
        :param num_episodes: number of episodes to train the model
        """
        num_steps = cum_reward = max_reward_train = 0
        episode_reward = []
        episode_loss = []

        for episode in range(num_episodes):

            state = self.em.reset()
            cum_loss = 0

            for time_step in count():

                num_steps += 1
                action = self.select_action(state, self.em.get_legal_actions())
                next_state, reward, done, _ = self.em.step(action)
                reward = torch.tensor([reward], device=self.device)
                self.memory.push(project_utils.Experience(state, action, next_state, reward, done))
                state = next_state
                cum_reward += reward.item()

                if self.memory.can_provide_sample(BATCH_SIZE):
                    cum_loss = self._train_step(cum_loss)

                if num_steps % TARGET_UPDATE == 0:
                    # Update the target net
                    self.target_net.load_state_dict(self.policy_net.state_dict())

                if self.em.done:
                    max_reward_train = max(max_reward_train, cum_reward)
                    episode_reward.append(cum_reward)
                    mean_loss = cum_loss / (time_step + 1)
                    episode_loss.append(mean_loss)
                    project_utils.plot(episode_reward, 20, 'Reward')
                    project_utils.plot(episode_loss, 20, 'Loss')
                    if is_ipython: display.clear_output(wait=True)
                    cum_reward = 0
                    break

    def _train_step(self, cum_loss):

        # Sampling from memory a batch of experiences
        experiences, importance, indices = self.memory.sample(BATCH_SIZE, priority_scale=self.scale)
        states, actions, rewards, next_states, dones = project_utils.extract_tensors(experiences, self.device)

        # Get the Q-values and the target values from the experiences
        current_q_values = QValues.get_current(self.policy_net, states, actions)
        next_q_values = QValues.get_next(self.target_net, next_states)
        target_q_values = (next_q_values * GAMMA) * (1 - dones).squeeze(1) + rewards
        weights = importance ** (1 - self.rate)
        errors, loss = project_utils.get_loss(current_q_values, target_q_values, weights, BATCH_SIZE, self.device)

        # Update the priorities
        self.memory.set_priorities(indices, errors)

        # Finish the training step
        cum_loss += loss.item()
        self.optimizer.zero_grad()
        loss.backward()

        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return cum_loss
