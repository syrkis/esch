# %% main.py
# example usage of esch
# by: Noah Syrkis

# %% from jax import random
import esch
import numpy as np


# %%
def main():
    """Example usage of esch plotting."""
    # Generate some data

    # Static plot
    matrix = np.random.randn(3, 100, 100)
    left = esch.EdgeConfig(
        ticks=[(0, "a"), (1, "b")],
        show_on="all",
    )
    top = esch.EdgeConfig(label=["a", "b", "c"], show_on="all", ticks=[(0, "a"), (1, "b")])
    edge = esch.EdgeConfigs(left=left, top=top)
    drawing = esch.plot(
        matrix,
        # animated=True,
        edge=edge,
        path="noah.svg",
    )

    return drawing


main()
# plt.show()
# print(anim.figure)
