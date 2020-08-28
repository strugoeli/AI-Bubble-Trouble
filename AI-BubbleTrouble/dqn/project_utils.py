import math

import torch
import numpy as np
from collections import namedtuple
import torch.nn.functional as F
import matplotlib.pyplot as plt

Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward', 'done'))


def init_networks(num_channels, num_actions, model, device):
    """
    :param num_channels: Number of channels in the input
    :param num_actions: Number of outputs
    :param model: The network
    :param device: Current device
    :return: initialized policy and target networks
    """
    policy_net = model(num_channels, num_actions).to(device)
    target_net = model(num_channels, num_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    return policy_net, target_net


def load_model(policy, target, optimizer, path):
    """
    Load trained model from the given path
    """
    checkpoint = torch.load(path, map_location={'cuda:0': 'cpu'})
    policy.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    target.load_state_dict(policy.state_dict())
    target.eval()


def get_state(obs):
    """
    Converting the given observation to Tensor and returns it
    """
    state = np.array(obs)
    state = state.transpose((2, 0, 1))
    state = torch.from_numpy(state)
    return state.unsqueeze(0)


def extract_tensors(experiences, device):
    """
    Extracts the given experiences into tuple of Tensors and returns it
    """
    # Convert batch of Experiences to Experience of batches
    batch = Experience(*zip(*experiences))
    actions_b = tuple((map(lambda a: torch.tensor([[a]], device=device), batch.action)))
    rewards_b = tuple((map(lambda r: torch.tensor([r], device=device), batch.reward)))
    dones_b = tuple((map(lambda d: torch.tensor([[float(d)]], device=device), batch.done)))

    t1 = torch.cat(batch.state).to(device)
    t2 = torch.cat(actions_b)
    t3 = torch.cat(rewards_b)
    t4 = torch.cat([s for s in batch.next_state if s is not None]).to(device)
    t5 = torch.cat(dones_b)

    return t1, t2, t3, t4, t5


def get_loss(current_q_values, target_q_values, importance, batch_size, device):
    """
    Returns the weighted loss by the given importance and the TD errors
    """
    loss = (torch.tensor(importance, device=device) * F.smooth_l1_loss(current_q_values,
                                                                       target_q_values.unsqueeze(1))).mean()
    errors = abs(target_q_values.unsqueeze(1) - current_q_values.detach()).cpu().numpy().reshape(batch_size)
    return errors, loss


def get_euclidian_closest_bubble(game, bubbles_list):
    """
    :return: euclidean distance of agent and closest bubble from bubbles_list
    """
    bubbles_dist = [euclidean_dist_bubble_and_player(bubble, game.players[0]) for bubble in bubbles_list]
    min_dist_bubble_index = int(np.argmin(np.array(bubbles_dist)))
    return bubbles_list[min_dist_bubble_index], bubbles_dist[min_dist_bubble_index]


def get_x_axis_closest_bubble(game, bubbles_list):
    """
    :return: X axis distance of agent and closest bubble from bubbles_list
    """
    bubbles_dist = [dist_from_bubble_and_player(bubble, game.players[0], axis=0) for bubble in bubbles_list]
    min_dist_bubble_index = int(np.argmin(np.array(bubbles_dist)))
    return bubbles_list[min_dist_bubble_index], bubbles_dist[min_dist_bubble_index]


def dist_from_bubble_and_player(bubble, player, axis=0):
    """
    :return: dist of player from bubble in axis
    """
    if axis == 0:
        player_spot = player.rect.centerx
        pos_bubble_spot = bubble.rect.left
        neg_bubble_spot = bubble.rect.right
        return min(abs(player_spot - pos_bubble_spot), abs(player_spot - neg_bubble_spot))
    else:
        return abs(bubble.rect.bottom - player.rect.top)


def euclidean_dist_bubble_and_player(bubble, player):
    """
    :return: euclidean dist of player from bubble
    """
    return math.sqrt(math.pow(dist_from_bubble_and_player(bubble, player, 0), 2) + math.pow(
        dist_from_bubble_and_player(bubble, player, 1), 2))


def plot(values, moving_avg_period, val_type):
    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel(val_type)
    plt.plot(values, label=val_type)
    moving_avg = get_moving_average(moving_avg_period, values)
    plt.plot(moving_avg, label='Moving Average')
    plt.legend()
    plt.pause(0.001)
    print("Episode", len(values), "\n", moving_avg_period, "episode moving avg:", moving_avg[-1])


def get_moving_average(period, values):
    values = torch.tensor(values, dtype=torch.float)
    if len(values) >= period:
        moving_avg = values.unfold(dimension=0, size=period, step=1) \
            .mean(dim=1).flatten(start_dim=0)
        moving_avg = torch.cat((torch.zeros(period - 1), moving_avg))
        return moving_avg.numpy()
    else:
        moving_avg = torch.zeros(len(values))
        return moving_avg.numpy()
