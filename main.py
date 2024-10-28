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
    plot(
        matrix,
        path="static.svg",
        xticks=[(0, "a"), (16, "b")],
        yticks=[(0, "ℙ"), (1, 2), (2, 3)],
        xlabel="time",
        ylabel="task",
    )

    # Animated plot
    tensor = random.normal(key, (32, 16, 32)).cumsum(axis=0)
    # tensor = prep(tensor)
    plot(
        tensor,
        animated=True,
        path="temp.svg",
        xlabel="time",
        ylabel="task",
        xticks=[(0, "a"), (16, "b")],
        yticks=[(0, "ℙ"), (1, 2), (2, 3)],
    )


if __name__ == "__main__":
    main()
