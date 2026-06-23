"""Shared figure-saving helper for quicklook plots."""

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import Normalize
from PIL import Image


def save_png(fig, path, dpi, compress_level):
    """Save `fig` to `path` as an RGB PNG at `dpi`.

    Renders the figure at `dpi`, then writes the Agg buffer's RGB channels
    straight through Pillow, skipping the constant (opaque) alpha channel that
    ``Figure.savefig`` would encode. Pixel-identical to
    ``savefig(path, dpi=dpi, facecolor="w")`` at the same resolution, but ~30%
    faster to encode and a smaller file.

    `compress_level` (0-9, the PNG zlib level) is caller-supplied because the
    speed/size trade-off depends on the content: dense detector images (L0/L1)
    favour a low level (~1) for a big encode win at negligible size cost, while
    highly compressible line plots (L2) favour the default (~6), where a low
    level barely saves time but inflates the file by ~55%.

    `dpi` is applied here rather than at save time on purpose: the Agg buffer's
    pixel dimensions are fixed by the figure dpi at ``draw`` time, so it must be
    set before the draw or the image silently renders at the default dpi.
    """
    fig.set_dpi(dpi)
    fig.patch.set_facecolor("w")
    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    Image.fromarray(rgb, "RGB").save(
        path, format="png", dpi=(dpi, dpi), compress_level=compress_level
    )


def save_image_png(image, path, cmap, vmin, vmax, origin="lower", compress_level=1):
    """Save `image` directly as a native-size RGB PNG.

    This bypasses matplotlib's figure rasterization so the output dimensions are
    exactly one PNG pixel per input array pixel. `origin="lower"` mirrors the
    display orientation used by `imshow(..., origin="lower")`.
    """
    arr = np.asarray(image)
    if origin == "lower":
        arr = np.flipud(arr)
    elif origin != "upper":
        raise ValueError(f"origin must be 'lower' or 'upper'; got {origin!r}")

    masked = np.ma.masked_invalid(arr)
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap_obj = colormaps[cmap].copy()
    cmap_obj.set_bad("black")
    rgb = cmap_obj(norm(masked), bytes=True)[..., :3]
    Image.fromarray(rgb, "RGB").save(
        path, format="png", compress_level=compress_level
    )
