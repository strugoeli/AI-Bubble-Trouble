from genetic.gui import start_main_menu
from genetic.gui import start_level
from genetic.gui import init_gui_normal
from genetic.gui import init_gui_train
from genetic.game import *
import sys
import datetime
import time
import pickle
#import visualize
import neat


NUM_OF_GENERATIONS = 150

def eval_genomes(genomes,config):
    nets = []
    ge = []
    for g_id,g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g,config)
        nets.append(net)
        g.fitness = 0
        ge.append(g)
    for j in range(len(nets)):
        game, font, clock, screen, main_menu, load_level_menu = init_gui_train()
        start_level(1,game, font, clock, screen, main_menu, load_level_menu, train_mode=True, model = nets[j])
        game_score = game.get_score()
        current_fitness = game.score
        ge[j].fitness = current_fitness

def run_from_scratch(config_file,num_of_generations):

    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet,neat.DefaultStagnation,config_file)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))
    winner = p.run(eval_genomes,num_of_generations)
    with open("winner.pkl", "wb") as f:
        pickle.dump(winner, f)
        f.close()
    visualize.draw_net(config, winner, True)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

def run_from_existing(checkpoint_filename,num_of_generations,winner_output_file):
    p = neat.Checkpointer.restore_checkpoint(checkpoint_filename)
    #p.add_reporter(genetic.StdOutReporter(True))
    #stats = genetic.StatisticsReporter()
    #p.add_reporter(stats)
    #p.add_reporter(genetic.Checkpointer(5,300,"existing-checkpoint"))

    winner = p.run(eval_genomes,num_of_generations)
    with open(winner_output_file, "wb") as f:
        pickle.dump(winner, f)
        f.close()

def replay_genome(config_path, genome_path,model_num):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                                neat.DefaultStagnation, config_path)
    # Load requried NEAT config
    # Unpickle saved winner
    with open(genome_path, "rb") as f:
        genome = pickle.load(f)

    model = neat.nn.FeedForwardNetwork.create(genome, config)
    # Convert loaded genome into required data structure
    game, font, clock, screen, main_menu, load_level_menu = init_gui_train()
    # Call game with only the loaded genome
    start_level(1, game, font, clock, screen, main_menu, load_level_menu, train_mode=True, model=model, model_num = model_num)

"""
import visualize
def draw_genome(config_path,genome_path):
    # Load requried NEAT config
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                                neat.DefaultStagnation, config_path)
    # Unpickle saved winner
    with open(genome_path, "rb") as f:
        winner = pickle.load(f)
    visualize.draw_net(config, winner, True)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)
"""

def run_lite_agent():
    local_dir = os.path.dirname(__file__)
    config_path = local_dir + "/config-feedforward.txt"
    replay_genome(config_path,r"{}/best3-winner.pkl".format(os.path.dirname(__file__)),2)
    # replay_genome(config_path,r"best3-winner.pkl",model_num)
    # draw_genome(config_path,r"best3-winner.pkl")

def run_large_agent():
    local_dir = os.path.dirname(__file__)
    config_path = local_dir + "/config-feedforward.txt"
    replay_genome(config_path,r"{}/14000winner-final.pkl".format(os.path.dirname(__file__)),1)

if __name__ == '__main__':
    run_lite_agent()
#
# #original main
# if __name__ == '__main__':
#     local_dir = os.path.dirname(__file__)
#     config_path = local_dir + "/config-feedforward.txt"
#
#     #RUN FROM SCRATCH
#     #run_from_scratch(config_path,NUM_OF_GENERATIONS)
#
#     #RUN FROM EXISTING CHECKPOINT
#     #run_from_existing(r'C:\Users\rbensadoun\PycharmProjects\BubbleNEAT\best2-checkpoint149',1,r"C:\Users\rbensadoun\PycharmProjects\BubbleNEAT\best2winner")
#
#     # model_num = 1
#     #TO REPLAY A SPECIFIC WINNER
#     # replay_genome(config_path,r"best3-winner.pkl",model_num)
#     # draw_genome(config_path,r"best3-winner.pkl")
#
#     #game, font, clock, screen, main_menu, load_level_menu = init_gui_train()
#     # Call game with only the loaded genome
#     #start_level(4,     game, font, clock, screen, main_menu, load_level_menu)
#
#
#     def get_csv_columns():
#         result = []
#         cols = ['Popped', 'TimeCompleted', 'LivesLeft']
#         for col in cols:
#             for i in range(1, 6):
#                 result.append('{}Lvl{}'.format(col, i))
#         return result
#
#
#     def get_row_from_dicts(arr_of_dicts):
#         row = []
#         for dict in arr_of_dicts:
#             for i in range(1, 6):
#                 row.append(dict[i])
#         return row
#
#     #
#     # if __name__ == '__main__':
#     #     # PLAYER FUNCTIONALITIES
#     #     RESULTS_CSV_PATH = os.path.join(os.getcwd(), 'results.csv')
#     #
#     #     if not os.path.exists(RESULTS_CSV_PATH):
#     #         with open(RESULTS_CSV_PATH, "w+") as f:
#     #             csvwriter = csv.writer(f, delimiter=',')
#     #             csvwriter.writerow(['player_id'] + get_csv_columns())
#     #             pass
#     #
#     #     game, font, clock, screen, main_menu, load_level_menu = init_gui_train()
#     #     start_level(1, game, font, clock, screen, main_menu, load_level_menu)
#     #
#     #     # UNCOMMENT AFTER PLAYER HAS WARMED UP AND LEARNED THE GAME
#     #
#     #     # # ADD 1 TO PLAYER ID FOR EACH DIFFERENT PLAYER
#     #     PLAYER_ID = 1
#     #
#     #     with open(RESULTS_CSV_PATH, 'a', newline='\n', encoding='utf-8') as f:
#     #         csvwriter = csv.writer(f, delimiter=',')
#     #         csvwriter.writerow(
#     #             [PLAYER_ID] + get_row_from_dicts([game.popped_by_level, game.time_by_level, game.lives_at_level]))
