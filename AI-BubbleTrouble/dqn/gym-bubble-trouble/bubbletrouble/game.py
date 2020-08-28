from threading import Timer
import json

from bubbles import *
from player import *


class BubbleTroubleGame:

    def __init__(self, level=1):
        self.balls = []
        self.hexagons = []
        self.players = [Player()]
        self.player = self.players[0]
        self.bonuses = []
        self.level = level
        self.game_over = False
        self.level_completed = False
        self.is_running = True
        self.is_completed = False
        self.max_level = MAX_LEVEL
        self.is_multiplayer = False
        self.is_restarted = False
        self.dead_player = False
        self.mode = 'Classic'
        self.score = 0
        self.timers = None

        self.time_by_level = {i: 0 for i in range(1, 6)}
        self.popped_by_level = {i: 0 for i in range(1, 6)}
        self.lives_by_level = {i: 0 for i in range(1, 6)}

        self.num_of_popped = 0
        self.num_of_frames = 0

        with open(APP_PATH + 'max_level_available', 'r') as max_completed_level_file:
            max_level_available = max_completed_level_file.read()
            if max_level_available:
                self.max_level_available = int(max_level_available)
            else:
                self.max_level_available = 1

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
            player.set_position((WINDOWWIDTH / (num_of_players + 1)) * player_number)
            player.is_alive = True
        self.level_completed = False
        self.level = level
        if self.level > self.max_level_available:
            self.level = 1
            self.max_level_available = self.level

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
        self._set_timers()

    def exit_game(self):
        self._stop_timers()

    def _stop_timers(self):
        if self.timers is not None:
            for timer in self.timers:
                if timer.is_alive():
                    timer.cancel()
        self.timers = None

    def move_player(self, direction):
        self.player.moving_right = direction == -1
        self.player.moving_left = direction == 1

    def fire_player(self):
        self.move_player(0)
        self.player.shoot()

    def stop_player(self):
        self.player.stop_moving()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exit_game()

    def _start_timer(self):
        self._timer(1, self._tick_second, self.time_left)

    def _set_timers(self):
        self._stop_timers()
        self.timers = [Timer(t, self._tick, []) for t in range(0, self.time_left)]
        for timer in self.timers:
            timer.start()

    def _tick(self):
        self.time_left -= 1
        if self.time_left == 0:
            self._decrease_lives(self.player)

    def _check_for_collisions(self):
        for player in self.players:
            self._check_for_bubble_collision(self.balls, True, player)
            self._check_for_bubble_collision(self.hexagons, False, player)

    def _check_for_bubble_collision(self, bubbles, is_ball, player):
        for bubble_index, bubble in enumerate(bubbles):
            if self._check_weapon_ball_collision(bubble):
                player.weapon.is_active = False
                self.score += 1
                if is_ball:
                    self._split_ball(bubble_index)
                else:
                    self._split_hexagon(bubble_index)
                self.num_of_popped += 1
                return True
            if pygame.sprite.collide_mask(bubble, player):
                player.is_alive = False
                self._decrease_lives(player)
                return True
        return False

    def _check_weapon_ball_collision(self, bubble):
        player = self.player
        return pygame.sprite.collide_rect(bubble, player.weapon) and player.weapon.is_active

    def _decrease_lives(self, player):
        player.lives -= 1
        if player.lives:
            self.dead_player = True
            player.is_alive = False
        else:
            self.game_over = True

    def _restart(self):
        self.load_level(self.level)

    def _split_ball(self, ball_index):
        ball = self.balls[ball_index]
        if ball.size > 1:
            self.balls.append(Ball(ball.rect.left - ball.size ** 2, ball.rect.top - 10, ball.size - 1, [-3, -5]))
            self.balls.append(Ball(ball.rect.left + ball.size ** 2, ball.rect.top - 10, ball.size - 1, [3, -5]))
        del self.balls[ball_index]

    def _split_hexagon(self, hex_index):
        hexagon = self.hexagons[hex_index]
        if hexagon.size > 1:
            self.hexagons.append(Hexagon(hexagon.rect.left, hexagon.rect.centery, hexagon.size - 1, [-3, -5]))
            self.hexagons.append(Hexagon(hexagon.rect.right, hexagon.rect.centery, hexagon.size - 1, [3, -5]))
        del self.hexagons[hex_index]

    def update(self):
        self.num_of_frames += 1
        if self.level_completed and not self.is_completed:
            self.time_by_level[self.level] = self.num_of_frames / FPS
            self.popped_by_level[self.level] = self.num_of_popped
            self.num_of_popped = 0
            self.num_of_frames = 0
            self.lives_by_level[self.level] = self.players[0].lives
            print(self.players[0].lives)
            self.load_level(self.level + 1)
        if self.game_over:
            self.popped_by_level[self.level] = self.num_of_popped
            print(self.players[0].lives)
            self.num_of_popped = 0
            self.is_running = False
        if self.dead_player:
            self.popped_by_level[self.level] = self.num_of_popped
            print(self.players[0].lives)
            self.num_of_popped = 0
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
            print(self.players[0].lives)
            self.level_completed = True
            if self.level == self.max_level:
                self.is_completed = True

    def _timer(self, interval, worker_func, iterations=0):
        if iterations and not self.dead_player and not self.level_completed and not self.is_restarted:
            Timer(interval, self._timer, [interval, worker_func, 0 if iterations == 0 else iterations - 1]).start()
            worker_func()

    def _tick_second(self):
        self.time_left -= 1
        if self.time_left == 0:
            for player in self.players:
                self._decrease_lives(player)
