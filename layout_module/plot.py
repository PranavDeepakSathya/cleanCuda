import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from .layout import Layout


def plot_layout(L, title=None, ax=None, cmap="viridis"):
    if L.rank != 2:
        print(f"layout rank is {L.rank}, not plotting")
        return

    M = L.get_mode(0).size
    N = L.get_mode(1).size

    grid = np.zeros((M, N), dtype=int)
    for i in range(M):
        for j in range(N):
            grid[i, j] = L((i, j))

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(4, N * 0.6), max(4, M * 0.6)))

    vmin, vmax = grid.min(), grid.max()
    ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")

    for i in range(M):
        for j in range(N):
            v = grid[i, j]
            norm = (v - vmin) / max(vmax - vmin, 1)
            color = "white" if norm < 0.55 else "black"
            ax.text(j, i, str(v), ha="center", va="center",
                    color=color, fontsize=10, fontweight="bold")

    _draw_mode_brackets(ax, L.shape.get_mode(0), axis=0, M=M, N=N)
    _draw_mode_brackets(ax, L.shape.get_mode(1), axis=1, M=M, N=N)

    ax.set_xticks(np.arange(N))
    ax.set_yticks(np.arange(M))
    ax.set_xticks(np.arange(-0.5, N, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, M, 1), minor=True)
    ax.grid(which="minor", color="gray", linewidth=0.5, alpha=0.3)
    ax.tick_params(which="minor", length=0)

    shape_str = _fmt_nested(L.shape)
    stride_str = _fmt_nested(L.stride)
    header = f"{shape_str} : {stride_str}"
    ax.set_title(f"{title}\n{header}" if title else header, fontsize=11)

    ax.set_xlabel("mode 1")
    ax.set_ylabel("mode 0")

    plt.tight_layout()
    return ax


def _draw_mode_brackets(ax, mode_shape, axis, M, N, depth=0, offset=0):
    if mode_shape.prof.is_atom():
        return
    sizes = [mode_shape.get_mode(i).size for i in range(mode_shape.rank)]
    pos = offset
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for i, sz in enumerate(sizes):
        color = colors[(depth * 3 + i) % 10]
        lw = 2.5 - depth * 0.6
        if axis == 0:
            y0 = pos - 0.5
            y1 = pos + sz - 0.5
            x = -0.6 - depth * 0.25
            ax.plot([x, x], [y0, y1], color=color, linewidth=lw,
                    clip_on=False, solid_capstyle="round")
        else:
            x0 = pos - 0.5
            x1 = pos + sz - 0.5
            y = M - 0.4 + depth * 0.25
            ax.plot([x0, x1], [y, y], color=color, linewidth=lw,
                    clip_on=False, solid_capstyle="round")
        _draw_mode_brackets(ax, mode_shape.get_mode(i), axis, M, N,
                            depth + 1, pos)
        pos += sz


def _fmt_nested(nt):
    if nt.prof.is_atom():
        return str(nt.int_tuple[0])
    parts = [_fmt_nested(nt.get_mode(i)) for i in range(nt.rank)]
    return "(" + ",".join(parts) + ")"