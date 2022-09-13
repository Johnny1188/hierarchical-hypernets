import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", dest="config", type=str, help="Config file to use")
    parser.add_argument("--wandb", action="store_true", help="Run with wandb logging")
    parser.add_argument("-desc", "--description", dest="description", type=str, help="A description of the run")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--solver", type=str, help="Target network to use (solver)")
    parser.add_argument("--data", type=str, help="Data to use (collection of tasks in case of continual learning)")
    return parser.parse_args()
