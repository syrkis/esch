from esch.plot import plot
from esch.edge import EdgeConfigs, EdgeConfig
import numpy as np


def main():
    """Example usage of esch plotting."""
    # Generate some data
    matrix = np.abs(np.random.randn(11, 11)).cumsum(axis=1)

    # Correct LaTeX syntax with single `$` for inline math
    left = EdgeConfig(ticks=[(i, "voxel " + str(i) + "Δ") for i in range(11)], show_on="all")
    bottom = EdgeConfig(label="Epochs", show_on="all", ticks=[(0, "a"), (1, "b")])
    edge = EdgeConfigs(left=left, bottom=bottom)  # right=right)

    drawing = plot(
        matrix,
        # animated=True,
        edge=edge,
        path="noah.svg",
    )

    return drawing


main()
