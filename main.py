# main.py
# example usage of esch
# by: Noah Syrkis

from jax import random
from esch import plot, prep


def main():
    """Example usage of esch plotting."""
    # Generate some data
    key = random.PRNGKey(0)

    # Static plot
    matrix = random.normal(key, (16, 32))
    plot(matrix, path="static.svg", xticks=["a", "b", "c"], yticks=["d", "e", "f"], show_ticks=True)

    # Animated plot
    tensor = random.normal(key, (100, 32, 64)).cumsum(axis=0)
    tensor = prep(tensor)
    # plot(tensor, animated=True, path="temp.svg")


if __name__ == "__main__":
    main()
