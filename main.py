# %% main.py
# example usage of esch
# by: Noah Syrkis

# %% from jax import random
from esch import plot, prep
import numpy as np
from IPython.display import SVG, HTML, display


# %%
def main():
    """Example usage of esch plotting."""
    # Generate some data

    # Static plot
    matrix = np.random.randn(16, 32)
    drawing = plot(
        matrix,
        # path="static.svg",
        xticks=[(0, "a"), (16, "b")],
        yticks=[(37, "ℙ"), (1, 2), (2, 3)],
        xlabel="time",
        ylabel="task",
    )

    # Animated plot
    tensor = np.random.randn(1001, 113, 113)
    anim = plot(
        tensor,
        animated=True,
        path="temp.svg",
        xlabel="time",
        ylabel="task",
        xticks=[(0, "a"), (16, "b")],
        yticks=[(36, "ℙ"), (1, 2), (2, 3)],
    )

    return drawing


svg = main()
