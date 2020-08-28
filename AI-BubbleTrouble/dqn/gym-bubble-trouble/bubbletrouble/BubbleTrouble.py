import pygame
from pygame.locals import K_LEFT, K_RIGHT, K_SPACE, K_ESCAPE, KEYUP, KEYDOWN, QUIT
from game import BubbleTroubleGame
from settings import *

pygame.init()
surface = pygame.Surface((WINDOWWIDTH, WINDOWHEIGHT))
clock = pygame.time.Clock()
font = pygame.font.SysFont('monospace', 30)
game = BubbleTroubleGame()
exit_game = False
screen = None
key_map = {0: K_LEFT, 1: K_RIGHT, 2: K_SPACE, 3: None}


def setup():
    pygame.display.set_caption('Bubble Trouble')
    pygame.mouse.set_visible(True)
    global screen
    screen = pygame.display.set_mode((WINDOWWIDTH, WINDOWHEIGHT))


def score():
    return game.score


def is_over():
    return game.game_over


def is_completed():
    return game.level_completed


def surface_image():
    return pygame.surfarray.array3d(surface)


def game_start(rand=True, timed=True):
    # game.restart()
    game.load_level(level=4)


def game_update(restart=True):
    game.update()
    draw_world()


def game_render():
    screen.blit(surface, (0, 0))
    pygame.display.update()


def game_loop():
    game_start()
    while True:
        game_update()
        handle_game_event()
        if exit_game:
            break
        game_render()
        clock.tick(70)


def quit_game():
    global exit_game
    exit_game = True
    pygame.display.quit()
    pygame.quit()
    game.exit_game()


def draw_ball(ball):
    surface.blit(ball.image, ball.rect)


def draw_hex(hexagon):
    surface.blit(hexagon.image, hexagon.rect)


def draw_player(player):
    surface.blit(player.image, player.rect)


def draw_weapon(weapon):
    surface.blit(weapon.image, weapon.rect)


def draw_timer():
    timer = font.render(str(game.time_left), 1, RED)
    rect = timer.get_rect()
    rect.bottomleft = 10, WINDOWHEIGHT - 10
    surface.blit(timer, rect)
    del timer


def draw_player_lives(player):
    player_image = pygame.transform.scale(player.image, (20, 20))
    rect = player_image.get_rect()
    for life_num in range(player.lives):
        surface.blit(
            player_image,
            (WINDOWWIDTH - (life_num + 1) * 20 - rect.width, 10)
        )


def draw_world():
    surface.fill(WHITE)
    for hexagon in game.hexagons:
        draw_hex(hexagon)
    for ball in game.balls:
        draw_ball(ball)
    if game.player.weapon.is_active:
        draw_weapon(game.player.weapon)
    draw_player(game.player)
    draw_player_lives(game.player)
    draw_timer()


def handle_key(key, down):
    if key not in key_map.values():
        print('Wrong control. Use left, right arrows to move, space to shoot and ESC to exit')
        return
    if down:
        if key == K_LEFT:
            game.move_player(direction=1)
        elif key == K_RIGHT:
            game.move_player(direction=-1)
        elif key == K_SPACE:
            game.fire_player()
        elif key is None:
            game.stop_player()  # Do nothing
    else:
        if key == K_LEFT or key == K_RIGHT:
            game.stop_player()


def handle_game_event():
    for event in pygame.event.get():
        if event.type != KEYDOWN and event.type != KEYUP:
            continue
        if event.type == QUIT or event.key == K_ESCAPE:
            quit_game()
            return
        handle_key(event.key, event.type == KEYDOWN)


if __name__ == '__main__':
    setup()
    game_loop()
