import torch
from manager import EnvManager
from strategy import EpsilonGreedyStrategy
from agent import Agent
from models import CNN
import torch.optim as optim
import project_utils as utils
from parameters import *
import os

N_EPISODES_TEST = 1
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
PATH_TO_MODEL = r'{}\trained_models\test2.7'


def run_dqn_agent(to_train=False):
    """
    Loading and testing an agent for the Bubble Trouble game

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    em = EnvManager(device, NUM_OF_CHANNELS, SKIP_FRAMES, to_skip=False, ep_live=True)
    num_actions = em.num_actions_available()
    test_em = EnvManager(device, NUM_OF_CHANNELS, SKIP_FRAMES, to_skip=False, ep_live=False)
    policy_net, target_net = utils.init_networks(NUM_OF_CHANNELS, num_actions, CNN, device)
    optimizer = optim.Adam(params=policy_net.parameters(), lr=LR)
    strategy = EpsilonGreedyStrategy(EPS_START, EPS_END, EPS_DECAY)

    if to_train:
        agent = Agent(policy_net, target_net, strategy, em, test_em, em.num_actions_available(), optimizer)
        agent.train(NUM_EPISODES)
    else:
        utils.load_model(policy_net, target_net, optimizer, PATH_TO_MODEL.format(os.path.dirname(__file__)))
        policy_net.eval()
        agent = Agent(policy_net, target_net, strategy, em, test_em, em.num_actions_available(), optimizer)
        agent.test(N_EPISODES_TEST)
