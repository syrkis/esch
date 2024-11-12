from esch.plot import plot
from esch.edge import EdgeConfigs, EdgeConfig
import numpy as np


def main():
    """Example usage of esch plotting."""
    # Generate some data
    matrix = np.random.randn(3, 11, 11)

    # Correct LaTeX syntax with single `$` for inline math
    left = EdgeConfig(ticks=[(0, r"train"), (1.0, "$1$")], label=r"𝑥₀", show_on="all")
    top = EdgeConfig(ticks=[(0, "𝑥₀"), (1.0, "𝑥₀")], label=r"𝑥₀", show_on="all")
    bottom = EdgeConfig(label=["𝑥₀", "b", "c"], show_on="all", ticks=[(0, "a"), (1, "b")])
    # right = EdgeConfig(label="RIGHT ON MOTHERUFCKS", show_on="all", ticks=[(0, "a"), (1, "b")])
    edge = EdgeConfigs(left=left, top=top, bottom=bottom)  # right=right)

    drawing = plot(
        matrix,
        # animated=True,
        edge=edge,
        path="noah.svg",
    )

    return drawing


main()
