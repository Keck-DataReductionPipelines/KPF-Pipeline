"""Shared figure-saving helper for quicklook plots."""

import numpy as np
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
