import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence


def ring(
    seq: Sequence[Sequence] | Sequence | np.ndarray, f_name: Sequence[str] | str | None = None, ax=None, path=None
):
    # init and config
    print(seq)
    exit()
    ax_was_none = ax is None
    if ax is None:
        fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(10, 10), dpi=100)
    # limit should be from 0 to max of all ps
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["polar"].set_visible(False)

    # plot
    ps = ps if isinstance(ps[0], (Sequence, np.ndarray)) else [ps]  # type: ignore
    ax.plot([0, 0], "black", linewidth=1)
    for p in ps:
        ax.plot(p, p, "o", markersize=2, color="black")
    plt.tight_layout()
    plt.savefig(f"{path}/{f_name}.svg") if f_name else plt.show()
    if ax_was_none:
        plt.close()


def small_multiples(fnames, seqs, f_name, n_rows=2, n_cols=2):
    assert (
        len(fnames) == len(seqs) and len(fnames) >= n_rows * n_cols
    ), "fnames and seqs must be the same length and n_rows * n_cols"
    fig, axes = plt.subplots(n_rows, n_cols, subplot_kw=dict(polar=True), figsize=(n_cols * 5, n_rows * 5), dpi=100)
    for ax, fname, seqs in zip(axes.flat, fnames, seqs):  # type: ignore
        ring(seqs, fname, ax=ax)
    # tight
    # plt.tight_layout()
    plt.savefig(f"{f_name}.svg") if f_name else plt.show()  # test
