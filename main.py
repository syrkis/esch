import esch
import numpy as np
import matplotlib.pyplot as plt


def the_tile():
    x = np.random.randn(10, 20)
    esch.tile(x, path="paper/figs/the_2d_tile.svg")
    x = np.random.randn(3, 11, 10)
    esch.tile(x, path="paper/figs/the_3d_tile.svg")


def the_ring():
    x = np.random.randint(0, 10, size=(10,))
    esch.ring(x, path="paper/figs/the_1d_ring.svg")


def the_line():
    arr = np.random.randn(2, 50).cumsum(1)
    assert arr.ndim <= 2
    linestyles = ["-", "--", "-.", ":"]
    # for i, y in enumerate(arr):
    # x = np.arange(y.shape[-1])
    # plt.plot(x, y, patterns[i % len(patterns)], color="black")
    if arr.ndim == 1:
        arr = arr[np.newaxis, ...]
    for i, y in enumerate(arr):
        x = np.arange(y.shape[-1])
        plt.step(x, y, color="black", linestyle=linestyles[i % len(linestyles)])

    # x = np.tile(np.arange(arr.shape[-1]), arr.shape[-2] if arr.ndim > 1 else 1).reshape(arr.shape)
    # x = np.arange(arr.shape[-1])[None, :].repeat(arr.shape[-2], axis=0).reshape(arr.shape)
    # x = np.random.randn(10)
    # plt.step(x, arr, color="black")
    plt.savefig("paper/figs/the_line.svg")


if __name__ == "__main__":
    # the_ring()
    # the_tile()
    the_line()
