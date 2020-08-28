from wrappers import *
from env import BubbleTroubleEnv
from settings import *
from project_utils import get_state

TOO_LOW = 80
TOO_CLOSE = 50
TOO_FAR = 320
TOO_FAR_X = 150

ACTION_LEFT = 0
ACTION_RIGHT = 1
ACTION_FIRE = 2
ACTION_IDLE = 3

reward_dict = {'moving': .0, 'fire': .0, 'score': 1, 'death': -1., 'win': 0, 'step': .0}


class EnvManager:
    """
    This class is responsible for managing the Bubble trouble environment
    """

    def __init__(self, device, num_frames=4, skip=2, to_skip=True, ep_live=False):
        self.device = device
        self.env = BubbleTroubleEnv(rewards=reward_dict)
        self.env = FrameStack(self.env, num_frames)
        if to_skip:
            self.env = MaxAndSkipEnv(self.env, skip)
        if ep_live:
            self.env = EpisodicLifeEnv(self.env)
        self.done = False
        self.env.reset()
        self.curr_screen = None
        self.frame_counter = 0

    def reset(self):
        """
        Reset the environment and returns the first observation
        """
        ob = self.env.reset()
        ob = get_state(ob)
        return ob

    def step(self, action):
        """
        Execute a single step of the game

        :param action: int - action to execute
        :return: observation - Tensor, reward -int, done -boolean, info -Not in use
        """
        ob, reward, done, info = self.env.step(action)
        ob = get_state(ob)
        self.done = done
        return ob, reward, done, info

    def close(self):
        self.env.close()

    def render(self):
        """
        :return: current frame of the game
        """
        return self.env.render('rgb_array')

    def num_actions_available(self):
        """
        :return: Number of steps
        """
        return self.env.action_space.n

    def just_starting(self):
        """
        :return: True if is the first frame and false otherwise
        """
        return self.curr_screen is None

    def can_shoot(self):
        """
        :return: True if the player may shoot and false otherwise
        """
        return self._get_player().can_shoot()

    def get_legal_actions(self):
        """
        :return:  Returns set of legal action minus 'irrational' if exist
        """
        player = self._get_player()
        actions = []
        logic_actions = self.get_logic_actions()
        if logic_actions:
            return logic_actions
        if player.rect.left > 0:
            actions.append(ACTION_LEFT)
        if player.rect.right < WINDOWWIDTH:
            actions.append(ACTION_RIGHT)
        if player.can_shoot() and self.env.closest_dist < TOO_FAR_X:
            actions.append(ACTION_FIRE)
        actions.append(ACTION_IDLE)
        return actions

    def get_logic_actions(self):
        """
        :return: Returns 'logical' set of action that the player may execute
        """
        if not self.env.closest_ball_euc:
            return []

        player = self._get_player()
        ball_rec = self.env.closest_ball_euc.rect
        on_the_left = ball_rec.centerx <= player.rect.centerx

        if self._is_in_danger(ball_rec):

            if on_the_left:
                return [ACTION_RIGHT, ACTION_FIRE] if player.can_shoot() else [ACTION_RIGHT]
            else:
                return [ACTION_LEFT, ACTION_FIRE] if player.can_shoot() else [ACTION_LEFT]

        if self._is_too_far() and self._is_normal_ball():

            if on_the_left:
                return [ACTION_LEFT]
            else:
                return [ACTION_RIGHT]

    def _is_in_danger(self, ball_rec):
        # True if the player in danger zone and false otherwise
        too_low = WINDOWHEIGHT - ball_rec.centery <= TOO_LOW
        player = self._get_player()
        return self._is_too_close() and (not player.can_shoot() or player.can_shoot() and too_low)

    def _is_normal_ball(self):
        # Normal means not Hexagon
        return self.env.closest_ball_euc in self.env.game.balls

    def _is_too_close(self):
        # True if too close to the closest ball and false otherwise
        return self.env.closest_dist_euc < TOO_CLOSE

    def _is_too_far(self):
        # True if too far from the closest ball and false otherwise
        return self.env.closest_dist > TOO_FAR

    def _get_game(self):
        return self.env.game

    def _get_player(self):
        return self._get_game().player
