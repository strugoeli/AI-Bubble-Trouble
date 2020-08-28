import sys
from threading import Timer
import random
import json
import numpy as np
from genetic.bubbles import *
from genetic.player import *
from genetic.bonuses import *
from genetic.settings import *

class Game:

    def __init__(self, level=1):
        self.balls = []
        self.hexagons = []
        self.players = [Player()]
        self.bonuses = []

        self.positions_of_balls_last_frame = {}

        self.score = 0
        self.level = level
        self.num_of_shoots = 0
        self.game_over = False
        self.level_completed = False
        self.is_running = True
        self.is_completed = False
        self.max_level = MAX_LEVEL
        self.is_multiplayer = False
        self.is_restarted = False
        self.dead_player = False
        self.mode = 'Classic'

        self.is_ai = False
        self.fitness_penalty = 0

        with open(APP_PATH + 'max_level_available', 'r') as \
                max_completed_level_file:
            max_level_available = max_completed_level_file.read()
            if max_level_available:
                self.max_level_available = int(max_level_available)
            else:
                self.max_level_available = 1

    def update_positions_balls(self):
        balls = {ball : (ball.rect.centerx,ball.rect.centery) for ball in self.balls}
        hexagons = {hexa : (hexa.rect.centerx,hexa.rect.centery) for hexa in self.hexagons}
        for h in hexagons:
            balls[h] = hexagons[h]
        self.positions_of_balls_last_frame = balls


    def get_orientation_closest_ball(self):
        frame_closest_ball = self.get_closest_ball_to_player_x_axis()
        user_x_position = self.players[0].rect.centerx
        x_approaching = 0
        y_approaching = 0
        if frame_closest_ball and self.positions_of_balls_last_frame:
            try:
                position_last_frame = self.positions_of_balls_last_frame[frame_closest_ball]
                x_approaching = 0 if abs(frame_closest_ball.rect.centerx - user_x_position) > abs(
                    position_last_frame[0] - user_x_position) else 1
                y_approaching = 0 if frame_closest_ball.rect.centery > position_last_frame[1] else 1
            except:
                pass
        return x_approaching,y_approaching


    def get_closest_ball_to_player_x_axis(self):
        closest_ball = None
        closest_ball_distance = 0
        for ball in self.balls:
            user_dist_from_ball = ((ball.rect.centerx - self.players[0].rect.centerx)**2+(ball.rect.centery - self.players[0].rect.centery)**2)**0.5
            if user_dist_from_ball > closest_ball_distance:
                closest_ball_distance = user_dist_from_ball
                closest_ball = ball
        for hexagon in self.hexagons:
            user_dist_from_ball = ((hexagon.rect.centerx - self.players[0].rect.centerx)**2+(hexagon.rect.centery - self.players[0].rect.centery)**2)**0.5
            if user_dist_from_ball > closest_ball_distance:
                closest_ball_distance = user_dist_from_ball
                closest_ball = hexagon

        return closest_ball

    def get_closest_bonus_to_player_x_axis(self):
        closest_bonus = None
        closest_bonus_distance = 0
        for bonus in self.bonuses:
            user_dist_from_bonus = abs(bonus.rect.centerx - self.players[0].rect.centerx)
            if user_dist_from_bonus > closest_bonus_distance:
                closest_bonus_distance = user_dist_from_bonus
                closest_bonus = bonus
        return closest_bonus


    def load_level(self, level):
        self.is_restarted = True
        if self.is_multiplayer and len(self.players) == 1:
            self.players.append(Player('player2.png'))
        self.balls = []
        self.hexagons = []
        self.bonuses = []
        self.dead_player = False
        for index, player in enumerate(self.players):
            player_number = index + 1
            num_of_players = len(self.players)
            player.set_position(
                (WINDOWWIDTH / (num_of_players + 1)) * player_number
            )
            player.is_alive = True
        self.level_completed = False
        self.level = level
        if self.level > self.max_level_available:
            self.max_level_available = self.level
            with open(APP_PATH + 'max_level_available', 'w') as \
                    max_completed_level_file:
                max_completed_level_file.write(str(self.max_level_available))
        with open(APP_PATH + 'levels.json', 'r') as levels_file:
            levels = json.load(levels_file)
            level = levels[str(self.level)]
            self.time_left = level['time']
            for ball in level['balls']:
                x, y = ball['x'], ball['y']
                size = ball['size']
                speed = ball['speed']
                self.balls.append(Ball(x, y, size, speed))
            for hexagon in level['hexagons']:
                x, y = hexagon['x'], hexagon['y']
                size = hexagon['size']
                speed = hexagon['speed']
                self.hexagons.append(Hexagon(x, y, size, speed))
        self._start_timer()

    def get_time_left(self):
        """

        :return: time left to complete level
        """
        return int(np.ceil(self.time_left))


    def _start_timer(self):
        self._timer(1, self._tick_second, self.time_left)

    def _check_for_collisions(self):
        for player in self.players:
            self._check_for_bubble_collision(self.balls, True, player)
            self._check_for_bubble_collision(self.hexagons, False, player)
            self._check_for_bonus_collision(player)


    def add_to_score(self, to_add):
        """
        adds to_add to game's current score
        :param self:
        :param to_add: amount of points to be added to score
        :return:
        """
        self.score += to_add

    def get_score(self):
        """
        :param self:
        :return: game's current score
        """
        return int(self.score)

    def _check_for_bubble_collision(self, bubbles, is_ball, player):
        for bubble_index, bubble in enumerate(bubbles):
            if pygame.sprite.collide_rect(bubble, player.weapon) \
                    and player.weapon.is_active:
                self.add_to_score(50)
                player.weapon.is_active = False
                if is_ball:
                    self._split_ball(bubble_index)
                else:
                    self._split_hexagon(bubble_index)
                return True
            if pygame.sprite.collide_mask(bubble, player):
                player.is_alive = False
                self._decrease_lives(player)
                return True
        return False

    def _check_for_bonus_collision(self, player):
        for bonus_index, bonus in enumerate(self.bonuses):
            if pygame.sprite.collide_mask(bonus, player):
                self._activate_bonus(bonus.type, player)
                del self.bonuses[bonus_index]
                return True
        return False

    def _decrease_lives(self, player):
        player.lives -= 1
        if player.lives:
            self.dead_player = True
            player.is_alive = False
        else:
            self.game_over = True

    def _restart(self):
        self.load_level(self.level)

    @staticmethod
    def _drop_bonus():
        if random.randrange(BONUS_DROP_RATE) == 0:
            bonus_type = random.choice(bonus_types)
            return bonus_type

    def _activate_bonus(self, bonus, player):
        if bonus == BONUS_LIFE:
            player.lives += 1
        elif bonus == BONUS_TIME:
            self.time_left += 10

    def _split_ball(self, ball_index):
        ball = self.balls[ball_index]
        if ball.size > 1:
            self.balls.append(Ball(
                ball.rect.left - ball.size**2,
                ball.rect.top - 10, ball.size - 1, [-3, -5])
            )
            self.balls.append(
                Ball(ball.rect.left + ball.size**2,
                     ball.rect.top - 10, ball.size - 1, [3, -5])
            )
        del self.balls[ball_index]
        bonus_type = self._drop_bonus()
        if bonus_type:
            bonus = Bonus(ball.rect.centerx, ball.rect.centery, bonus_type)
            self.bonuses.append(bonus)

    def _split_hexagon(self, hex_index):
        hexagon = self.hexagons[hex_index]
        if hexagon.size > 1:
            self.hexagons.append(
                Hexagon(hexagon.rect.left, hexagon.rect.centery,
                        hexagon.size - 1, [-3, -5]))
            self.hexagons.append(
                Hexagon(hexagon.rect.right, hexagon.rect.centery,
                        hexagon.size - 1, [3, -5]))
        del self.hexagons[hex_index]
        bonus_type = self._drop_bonus()
        if bonus_type:
            bonus = Bonus(hexagon.rect.centerx, hexagon.rect.centery,
                          bonus_type)
            self.bonuses.append(bonus)

    def update(self):
        if self.level_completed:
            if  not self.is_completed:
                self.load_level(self.level + 1)
                self.add_to_score(TIME_LEFT_SCORE_FACTOR * self.get_time_left())
            else:
                self.is_running = False

        if self.game_over:
            self.is_running = False
            #pygame.quit()
            #sys.exit()
        if self.dead_player:
            self._restart()
        self._check_for_collisions()
        for ball in self.balls:
            ball.update()
        for hexagon in self.hexagons:
            hexagon.update()
        for player in self.players:
            player.update()
        for bonus in self.bonuses:
            bonus.update()
        if not self.balls and not self.hexagons:
            self.level_completed = True
            if self.level == self.max_level_available:
                self.is_completed = True

    def _timer(self, interval, worker_func, iterations=0):
        if iterations and not self.dead_player and not \
                self.level_completed and not self.is_restarted:
            Timer(

                interval, self._timer,
                [interval, worker_func, 0 if iterations ==
                    0 else iterations - 1]
            ).start()
            worker_func()

    def _tick_second(self):
        self.time_left -= 1
        if self.time_left == 0:
            for player in self.players:
                self._decrease_lives(player)
