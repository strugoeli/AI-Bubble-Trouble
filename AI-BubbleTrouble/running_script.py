import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def install_packages():
    install("pygame")
    install("neat-python")
    install("torch")
    install("gym")
    install("matplotlib")

install_packages()

from dqn.main import run_dqn_agent
from genetic.main import run_large_agent, run_lite_agent

def main():

    print("Welcome to Bubble Trouble AI navigator!")
    print("Which agent would you like to run?")
    print("1 - Deep Q-learning")
    print("2 - Neuroevolution ")
    print("Enter your choice please (1 or 2):")
    model_choice = int(input())
    if model_choice == 1:
        print("Which operation would you like to run?")
        print("1 - Test")
        print("2 - Train : Not recommended without gpu")
        print("Enter your choice please (1 or 2):")
        to_train = int(input()) - 1
        run_dqn_agent(to_train)
    elif model_choice == 2:
        print("Fine, you want to run the NEAT model, but which one?")
        print("1 - Lite")
        print("2 - Large")
        print("Enter your choice please (1 or 2):")
        neat_choice = int(input())
        if neat_choice == 1:
            run_lite_agent()
        elif neat_choice == 2:
            run_large_agent()


if __name__ == '__main__':
    main()
