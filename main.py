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
    matrix = np.random.randn(10, 6, 32)
    left = esch.EdgeConfig(ticks=[(0, "a"), (24, "b")], label="Time", show_on="first")
    edge = esch.EdgeConfigs(bottom=left)
    drawing = esch.plot(
        matrix,
        # animated=True,
        edge=edge,
        path="noah.svg",
    )

    return drawing


main()
plt.show()
# print(anim.figure)
