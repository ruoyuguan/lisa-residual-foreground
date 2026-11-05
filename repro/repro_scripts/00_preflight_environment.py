#!/usr/bin/env python
"""Check the Python environment used by the reproduction pipeline."""

from __future__ import annotations

import platform
import sys

import h5py
import matplotlib
import numpy
import pandas
import tables


def main() -> int:
    print(f"Python {sys.version.replace(chr(10), ' ')}")
    print(f"platform: {platform.platform()}")
    print(
        "packages: "
        f"numpy={numpy.__version__} "
        f"pandas={pandas.__version__} "
        f"h5py={h5py.__version__} "
        f"matplotlib={matplotlib.__version__} "
        f"tables={tables.__version__}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
