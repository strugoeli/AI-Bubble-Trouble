from pygame.locals import *
from collections import OrderedDict
import numpy as np
from genetic.game import *
from genetic.menu import *
from neat.math_util import softmax
from genetic.settings import *

def init_gui_train():
    """
    starts gui for game
    :return:
    """
    pygame.init()
    pygame.display.set_caption('Bubble Trouble')
    pygame.mouse.set_visible(True)
    screen = pygame.display.set_mode((WINDOWWIDTH, WINDOWHEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('monospace', 30)
    game = Game()

    main_menu = Menu(
        screen, OrderedDict(
            [('Single Player', start_single_player_level_menu),
                ('Two Players', start_multiplayer_level_menu),
                ('AI Player', start_ai_level_menu),
                ('Quit', quit_game)]
        )
    )
    def start_level_agent(level, game, font, clock, screen, main_menu, load_level_menu):
        return start_level(level, game, font, clock, screen, main_menu, load_level_menu)

    levels_available = [(str(lvl), (start_level_agent, lvl))
                        for lvl in range(1, game.max_level_available + 1)]
    levels_available.append(('Back', back))
    load_level_menu = Menu(screen, OrderedDict(levels_available))

    return game, font, clock, screen, main_menu, load_level_menu

def init_gui_normal():
    pygame.init()
    pygame.display.set_caption('Bubble Trouble')
    pygame.mouse.set_visible(True)
    screen = pygame.display.set_mode((WINDOWWIDTH, WINDOWHEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('monospace', 30)
    game = Game()

    main_menu = Menu(
        screen, OrderedDict(
            [('Single Player', start_single_player_level_menu),
             ('Two Players', start_multiplayer_level_menu),
             ('AI Player', start_ai_level_menu),
             ('Quit', quit_game)]
        )
    )
    levels_available = [(str(lvl), (start_level, lvl))
                        for lvl in range(1, game.max_level_available + 1)]
    levels_available.append(('Back', back))
    load_level_menu = Menu(screen, OrderedDict(levels_available))
    return game, font, clock, screen, main_menu, load_level_menu


MOVE_LEFT = 'left'
MOVE_RIGHT = 'right'
SHOOT = 'shoot'
ACTION_MAP = {0 : 'left', 1: 'right', 2:'shoot'}

def start_level(level, game, font, clock, screen, main_menu, load_level_menu,train_mode=False, model = None,model_num=None):
    game.load_level(level)
    main_menu.is_active = False
    pygame.mouse.set_visible(False)

    while game.is_running:
        game.update()
        draw_world(game, font, clock, screen, main_menu, load_level_menu)
        handle_game_event(game, font, clock, screen, main_menu, load_level_menu,model,train_mode,model_num)

        pygame.display.update()
        # if game.is_completed or game.game_over or \
        #         game.level_completed or game.is_restarted:
        #     pygame.time.delay(5)
        # if game.dead_player:
        #     pygame.time.delay(5)
        if game.is_restarted:
            game.is_restarted = False
            game._start_timer()
        clock.tick(FPS)


def start_main_menu(game, font, clock, screen, main_menu, load_level_menu):
    while main_menu.is_active:
        main_menu.draw()
        handle_menu_event(main_menu,game, font, clock, screen, main_menu, load_level_menu)
        pygame.display.update()
        clock.tick(FPS)


def start_load_level_menu(game, font, clock, screen, main_menu, load_level_menu):
    load_level_menu.is_active = True
    while load_level_menu.is_active:
        load_level_menu.draw()
        handle_menu_event(load_level_menu,game, font, clock, screen, main_menu, load_level_menu)
        pygame.display.update()
        clock.tick(FPS)


def start_single_player_level_menu(game, font, clock, screen, main_menu, load_level_menu):
    game.is_multiplayer = False
    start_load_level_menu(game, font, clock, screen, main_menu, load_level_menu)


def start_multiplayer_level_menu(game, font, clock, screen, main_menu, load_level_menu):
    game.is_multiplayer = True
    start_load_level_menu(game, font, clock, screen, main_menu, load_level_menu)

def start_ai_level_menu(game, font, clock, screen, main_menu, load_level_menu):
    game.is_multiplayer = False
    game.is_ai = True
    start_load_level_menu(game, font, clock, screen, main_menu, load_level_menu)

def quit_game(game, font, clock, screen, main_menu, load_level_menu):
    pygame.quit()
    sys.exit()


def back(game, font, clock, screen, main_menu, load_level_menu):
    """
    goes back to main menu
    :param game:
    :param font:
    :param clock:
    :param screen:
    :param main_menu:
    :param load_level_menu:
    :return:
    """
    load_level_menu.is_active = False


def draw_ball(ball, game, font, clock, screen, main_menu, load_level_menu):
    screen.blit(ball.image, ball.rect)


def draw_hex(hexagon, game, font, clock, screen, main_menu, load_level_menu):
    screen.blit(hexagon.image, hexagon.rect)


def draw_player(player, game, font, clock, screen, main_menu, load_level_menu):
    screen.blit(player.image, player.rect)


def draw_weapon(weapon, game, font, clock, screen, main_menu, load_level_menu):
    screen.blit(weapon.image, weapon.rect)


def draw_bonus(bonus, game, font, clock, screen, main_menu, load_level_menu):
    screen.blit(bonus.image, bonus.rect)


def draw_message(message, colour, game, font, clock, screen, main_menu, load_level_menu):
    label = font.render(message, 1, colour)
    rect = label.get_rect()
    rect.centerx = screen.get_rect().centerx
    rect.centery = screen.get_rect().centery
    screen.blit(label, rect)


def draw_timer( game, font, clock, screen, main_menu, load_level_menu):
    timer = font.render(str(game.time_left), 1, RED)
    rect = timer.get_rect()
    rect.bottomleft = 10, WINDOWHEIGHT - 10
    screen.blit(timer, rect)



def draw_players_lives(player, game, font, clock, screen, main_menu, load_level_menu, is_main_player=True):
    """
    draws players lives on screen
    :param player:
    :param game:
    :param font:
    :param clock:
    :param screen:
    :param main_menu:
    :param load_level_menu:
    :param is_main_player:
    :return:
    """
    player_image = pygame.transform.scale(player.image, (20, 20))
    rect = player_image.get_rect()
    for life_num in range(player.lives):
        if not is_main_player:
            screen.blit(player_image, ((life_num + 1) * 20, 10))
        else:
            screen.blit(
                player_image,
                (WINDOWWIDTH - (life_num + 1) * 20 - rect.width, 10)
            )

def draw_world( game, font, clock, screen, main_menu, load_level_menu):
    screen.fill(WHITE)
    for hexagon in game.hexagons:
        draw_hex(hexagon,game, font, clock, screen, main_menu, load_level_menu)
    for ball in game.balls:
        draw_ball(ball,game, font, clock, screen, main_menu, load_level_menu)
    for player_index, player in enumerate(game.players):
        if player.weapon.is_active:
            draw_weapon(player.weapon,game, font, clock, screen, main_menu, load_level_menu)
        draw_player(player,game, font, clock, screen, main_menu, load_level_menu)
        draw_players_lives(player,game, font, clock, screen, main_menu, load_level_menu)
    for bonus in game.bonuses:
        draw_bonus(bonus,game, font, clock, screen, main_menu, load_level_menu)
    draw_timer(game, font, clock, screen, main_menu, load_level_menu)
    if game.game_over:
        draw_message('Game over!', RED,game, font, clock, screen, main_menu, load_level_menu)
        start_main_menu(game, font, clock, screen, main_menu, load_level_menu)
    if game.is_completed:
        draw_message('Congratulations! You win!!!', PURPLE,game, font, clock, screen, main_menu, load_level_menu)
        #start_main_menu(game, font, clock, screen, main_menu, load_level_menu)
    if game.level_completed and not game.is_completed:
        draw_message('Well done! Level completed!', BLUE,game, font, clock, screen, main_menu, load_level_menu)
    if game.is_restarted:
        draw_message('Get ready!', BLUE,game, font, clock, screen, main_menu, load_level_menu)

def play_single_action(game, cur_action, player_num=0):
    """
    sets the proper boolean values so the player number player_num will perform the cur_action in the given game.
    :param game:
    :param cur_action:
    :param player_num:
    :return:
    """
    if cur_action == MOVE_LEFT:
        game.players[player_num].moving_left = True
        game.players[player_num].moving_right = False
    elif cur_action == MOVE_RIGHT:
        game.players[player_num].moving_right = True
        game.players[player_num].moving_left = False
    elif cur_action == SHOOT and not game.players[player_num].weapon.is_active:
        game.players[player_num].moving_left = False
        game.players[player_num].moving_right = False
        game.players[player_num].shoot()


def handle_ai_game_event(nn, game, font, clock, screen, main_menu, load_level_menu,model_num):
    user_x_position = game.players[0].rect.centerx
    closest_bonus = None
    closest_bonus_distance = 0
    closest_ball= game.get_closest_ball_to_player_x_axis()
    feature = (320,240) if not closest_ball else (closest_ball.rect.centerx,closest_ball.rect.centery)
    closest_ball_size = 0 if not closest_ball else closest_ball.size
    speed_x = 0 if not closest_ball else closest_ball.speed[0]
    speed_y = 0 if not closest_ball else closest_ball.speed[1]
    orientation_x,orientation_y = game.get_orientation_closest_ball()
    game.update_positions_balls()
    if model_num ==  1:
        net_output = nn.activate((feature[0],
                              feature[1],
                              user_x_position,
                              orientation_x,
                              orientation_y,
                              closest_ball_size,
                              speed_x,
                              speed_y,
                              len(game.balls)+len(game.hexagons)))
    elif model_num == 2:
        net_output = nn.activate((feature[0] / WINDOWWIDTH,
                                  feature[1] / WINDOWHEIGHT,
                                  user_x_position / WINDOWWIDTH,
                                  orientation_x,
                                  orientation_y,
                                  closest_ball_size / MAX_BALL_SIZE,
                                  speed_x / MAX_SPEED_X,
                                  speed_y / MAX_SPEED_Y,
                                  (len(game.balls) + len(game.hexagons)) / MAX_BALLS_AT_ALL_TIME))
        
    softmax_result = softmax(net_output)
    class_output = np.argmax(((softmax_result / np.max(softmax_result)) == 1).astype(int))
    play_single_action(game, ACTION_MAP[class_output])

def handle_game_event( game, font, clock, screen, main_menu, load_level_menu,model = None,train_mode = False,model_num = None):
    if (game.is_ai and model) or train_mode:
        handle_ai_game_event(model, game, font, clock, screen, main_menu, load_level_menu,model_num = model_num)
    else:
        for event in pygame.event.get():
            if event.type == KEYDOWN:
                if event.key == K_LEFT:
                    game.players[0].moving_left = True
                elif event.key == K_RIGHT:
                    game.players[0].moving_right = True
                elif event.key == K_SPACE and not game.players[0].weapon.is_active:
                    game.players[0].shoot()
                elif event.key == K_ESCAPE:
                    quit_game()
                if game.is_multiplayer:
                    if event.key == K_a:
                        game.players[1].moving_left = True
                    elif event.key == K_d:
                        game.players[1].moving_right = True
                    elif event.key == K_LCTRL and \
                            not game.players[1].weapon.is_active:
                        game.players[1].shoot()
            if event.type == KEYUP:
                if event.key == K_LEFT:
                    game.players[0].moving_left = False
                elif event.key == K_RIGHT:
                    game.players[0].moving_right = False
                if game.is_multiplayer:
                    if event.key == K_a:
                        game.players[1].moving_left = False
                    elif event.key == K_d:
                        game.players[1].moving_right = False
            if event.type == QUIT:
                quit_game()


def handle_menu_event(menu, game, font, clock, screen, main_menu, load_level_menu):
    """
    handles a selection in the menu
    :param menu:
    :param game:
    :param font:
    :param clock:
    :param screen:
    :param main_menu:
    :param load_level_menu:
    :return:
    """
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            quit_game(game, font, clock, screen, main_menu, load_level_menu)

        elif event.type == KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                if menu == main_menu:
                    quit_game(game, font, clock, screen, main_menu, load_level_menu)
                else:
                    start_main_menu(game, font, clock, screen, main_menu, load_level_menu)
            if (event.key == pygame.K_UP or event.key == pygame.K_DOWN)\
                    and menu.current_option is None:
                menu.current_option = 0
                pygame.mouse.set_visible(False)
            elif event.key == pygame.K_UP and menu.current_option > 0:
                menu.current_option -= 1
            elif event.key == pygame.K_UP and menu.current_option == 0:
                menu.current_option = len(menu.options) - 1
            elif event.key == pygame.K_DOWN \
                    and menu.current_option < len(menu.options) - 1:
                menu.current_option += 1
            elif event.key == pygame.K_DOWN \
                    and menu.current_option == len(menu.options) - 1:
                menu.current_option = 0
            elif event.key == pygame.K_RETURN and \
                    menu.current_option is not None:
                option = menu.options[menu.current_option]
                if not isinstance(option.function, tuple):
                    option.function()
                else:
                    option.function[0](option.function[1])

        elif event.type == MOUSEBUTTONUP:
            for option in menu.options:
                if option.is_selected:
                    if not isinstance(option.function, tuple):
                        option.function(game, font, clock, screen, main_menu, load_level_menu)
                    else:
                        option.function[0](option.function[1], game, font, clock, screen, main_menu, load_level_menu)

        if pygame.mouse.get_rel() != (0, 0):
            pygame.mouse.set_visible(True)
            menu.current_option = None
