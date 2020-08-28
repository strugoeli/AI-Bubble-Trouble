import gym
from settings import *
import numpy as np
import cv2 as cv
import random
import time
from bubbletrouble.game import BubbleTroubleGame
import pygame
from pygame.locals import K_LEFT, K_RIGHT, K_SPACE, K_ESCAPE, KEYUP, KEYDOWN, QUIT
from parameters import *
from project_utils import get_euclidian_closest_bubble, get_x_axis_closest_bubble

ACTION_LEFT = 0
ACTION_RIGHT = 1
ACTION_FIRE = 2
ACTION_IDLE = 3

T_LIMIT = 45
MAX_N_STEPS = FPS * T_LIMIT

key_map = {0: K_LEFT, 1: K_RIGHT, 2: K_SPACE, 3: None}
DEFAULT_REWARDS = {'moving': 0, 'fire': 0, 'score': 1, 'death': -1, 'win': 0, 'step': 0}

K = 1


class BubbleTroubleEnv(gym.Env):
    metadata = {'render.modes': ['rgb_array']}

    def __init__(self, rewards=None):
        pygame.init()
        self.rewards = rewards if rewards else DEFAULT_REWARDS
        self.action_space = gym.spaces.Discrete(4)
        self.n_steps = 0
        self.previous_score = None
        self.game = None
        self.closest_dist_euc, self.closest_ball_euc = 0, None
        self.closest_dist, self.closest_ball = 0, None

        # Init game
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('monospace', 30)
        self.screen = pygame.display.set_mode((WINDOWWIDTH, WINDOWHEIGHT), pygame.DOUBLEBUF)
        pygame.display.set_caption('Bubble Trouble')
        self.surface = pygame.Surface((WINDOWWIDTH, WINDOWHEIGHT), pygame.RESIZABLE).convert()

    def lives(self):
        """
        :return: Current number of lives of the player
        """
        return self.game.player.lives

    def can_shoot(self):
        """
        :return: True if the player can shoot and false otherwise
        """
        return self.game.player.can_shoot()

    def get_ball_list(self):
        """
        :return: List of all the ball objects in the game
        """
        return self.game.balls + self.game.hexagons

    def _update_closest_ball(self):
        # Update the closest ball and it's distance from the player
        # One for euclidean distance and once for x axis distance
        balls_list = self.get_ball_list()
        if balls_list:
            self.closest_ball, self.closest_dist = get_x_axis_closest_bubble(self.game, balls_list)
            self.closest_ball_euc, self.closest_dist_euc = get_euclidian_closest_bubble(self.game, balls_list)

    def reset(self):
        """
        Reset the environment and returns the first observation
        """
        self.n_steps = 0
        self.game = BubbleTroubleGame()
        self.game.load_level(1)
        self.previous_score = self.game.score
        return self._get_processed_screen()

    def step(self, action):
        """
        Execute a single step of the game
        :param action: int - action to execute
        """
        self.n_steps += 1
        self._make_single_step(action)

        state = self._get_processed_screen()
        win = self.game.is_completed
        done = self.is_done()
        reward = self._fitness(action, not self.game.player.is_alive, win, self._is_ball_hit())
        self.previous_score = self.game.score
        return state, reward, done, {}

    def is_done(self):
        """
        :return: True if the episode is over and false otherwise
        """
        return self.game.game_over or self.game.is_completed

    def render(self, mode='rgb_array', *args, **kwargs):
        """
        :return: current frame of the game
        """
        image = pygame.surfarray.array3d(self.screen)
        return image.swapaxes(1, 2).transpose((2, 0, 1))

    def _get_processed_screen(self):
        screen = self.render()
        screen = cv.cvtColor(screen, cv.COLOR_RGB2GRAY)
        screen = cv.resize(screen, (WIDTH, HEIGHT), interpolation=cv.INTER_AREA)
        screen = np.expand_dims(screen, -1)
        return screen

    def close(self):
        self.game.exit_game()

    def _fitness(self, action, dead, win, score_change):
        fitness = self.rewards['step']
        if action == ACTION_FIRE:
            fitness += self.rewards['fire']
        elif action != ACTION_IDLE:
            fitness += self.rewards['moving']
        if dead:
            fitness += self.rewards['death']
        if win:
            fitness += self.rewards['win']
        if score_change:
            fitness += self.rewards['score']
        return float(fitness)

    def _is_ball_hit(self):
        # True if an object is been destroyed and false otherwise
        return self.previous_score != self.game.score

    def _make_single_step(self, action):
        # Execute a single step of the game by drawing and updating the game image
        key = key_map[action]
        self.handle_key(key)
        self.game.update()
        self.clock.tick(FPS)
        pygame.display.update()
        self.draw_world()

    def render_with_states(self):

        img = np.ascontiguousarray(self.render(), dtype=np.uint8)
        c_x = int(self.game.player.position())
        closest_ball = self.closest_ball_euc.rect
        x, y = int(closest_ball.centerx), int(closest_ball.centery)
        p1, p2 = closest_ball.topleft, closest_ball.bottomright
        img = cv.rectangle(img, p1, p2, GREEN, 2)
        img = cv.line(img, (x, y), (c_x, WINDOWHEIGHT), GREEN, 1)
        return img

    def handle_key(self, key):
        game = self.game
        if key not in key_map.values():
            print('Wrong key')
            return
        if key == K_LEFT:
            game.move_player(direction=1)
        elif key == K_RIGHT:
            game.move_player(direction=-1)
        elif key == K_SPACE:
            game.fire_player()
        elif key is None:
            game.stop_player()

    def draw_world(self):
        self.surface.fill(WHITE)
        self.screen.blit(self.surface, (0, 0))
        for hexagon in self.game.hexagons:
            self.draw_hex(hexagon)
        for ball in self.game.balls:
            self.draw_ball(ball)
        self._update_closest_ball()
        if self.game.player.weapon.is_active:
            self.draw_weapon(self.game.player.weapon)
        self.draw_player(self.game.player)

    def draw_ball(self, ball):
        self.screen.blit(ball.image, ball.rect)

    def draw_hex(self, hexagon):
        self.screen.blit(hexagon.image, hexagon.rect)

    def draw_player(self, player):
        self.screen.blit(player.image, player.rect)

    def draw_weapon(self, weapon):
        self.screen.blit(weapon.image, weapon.rect)

    def draw_timer(self):
        timer = self.font.render(str(self.game.time_left), 1, RED)
        rect = timer.get_rect()
        rect.bottomleft = 10, WINDOWHEIGHT - 10
        self.screen.blit(timer, rect)
        del timer

    def draw_player_lives(self, player):
        player_image = pygame.transform.scale(player.image, (20, 20))
        rect = player_image.get_rect()
        y = 10
        for life_num in range(player.lives):
            x = WINDOWWIDTH - (life_num + 1) * 20 - rect.width
            self.screen.blit(player_image, (x, y))
