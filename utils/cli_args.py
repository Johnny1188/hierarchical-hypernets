import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb', action="store_true", help="Run with wandb logging")
    parser.add_argument('--epochs', type=int, help="Number of epochs")
    parser.add_argument('--solver', type=str, help="Target network to use (solver)")
    parser.add_argument('--data', type=str, help="Data to use (collection of tasks in case of continual learning)")
    return parser.parse_args()
