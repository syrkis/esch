# %% main.py
# example usage of esch
# by: Noah Syrkis

# %% from jax import random
import esch
import numpy as np
import matplotlib.pyplot as plt


# %%
def main():
    """Example usage of esch plotting."""
    # Generate some data

    # Static plot
    matrix = np.random.randn(3, 100, 20, 6)
    left = esch.EdgeConfig(ticks=[(0, "a"), (1, "b")], label="Time", show_on="first")
    top = esch.EdgeConfig(ticks=[(0, "a"), (3, "c")], label="Time", show_on="first")
    edge = esch.EdgeConfigs(left=left, top=top)
    drawing = esch.plot(
        matrix,
        animated=True,
        edge=edge,
        path="noah.svg",
    )

    return drawing


main()
# plt.show()
# print(anim.figure)
