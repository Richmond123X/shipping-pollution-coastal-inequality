# -*- coding: utf-8 -*-
"""Align Figure 3a with the established pastel palette used by panels b/c."""
from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
from PIL import Image


ROOT = Path(r"E:\transfer justice\results_v10_reconstructed_2019_2027")
SOURCE = ROOT / "figures" / "v10_new" / "figure_3" / "fig3a_increment_distribution_shares.png"
MANUSCRIPT_COPY = Path(
    r"E:\transfer justice\manuscript_materials_nature_sustainability"
    r"\01_results_shipping_network_and_empirical\figures"
    r"\fig3a_increment_distribution_shares.png"
)

# Emission bars use the same blue/pink pairing as Figure 3c; PWE bars use
# the green/purple pairing from Figure 3b.  Only the four categorical fills
# are changed, so the underlying chart geometry, text, and resolution remain
# untouched.
COLOR_MAP = {
    (224, 123, 112): (237, 166, 166),  # emission increase: established pink
    (108, 155, 207): (127, 181, 213),  # emission decrease: established blue
    (233, 184, 88): (185, 216, 198),   # PWE increase: established green
    (89, 170, 165): (205, 197, 223),   # PWE decrease: established purple
}


def main() -> None:
    with Image.open(SOURCE) as im:
        rgba = np.asarray(im.convert("RGBA")).copy()

    rgb = rgba[..., :3]
    replacements = {}
    for old, new in COLOR_MAP.items():
        mask = np.all(rgb == np.asarray(old, dtype=np.uint8), axis=2)
        replacements[old] = int(mask.sum())
        rgb[mask] = np.asarray(new, dtype=np.uint8)

    missing_targets = [
        new for new in COLOR_MAP.values()
        if not np.any(np.all(rgb == np.asarray(new, dtype=np.uint8), axis=2))
    ]
    if missing_targets:
        raise RuntimeError(f"Expected target colors are absent: {missing_targets}")

    Image.fromarray(rgba, mode="RGBA").save(SOURCE)
    MANUSCRIPT_COPY.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SOURCE, MANUSCRIPT_COPY)
    print(f"[complete] recolored={SOURCE} replacements={replacements}")
    print(f"[synced] manuscript_copy={MANUSCRIPT_COPY}")


if __name__ == "__main__":
    main()
