# -*- coding: utf-8 -*-
"""
The component plugins.

Importing this package registers every spec type it defines. `stock` holds
the components that map onto stock solph objects; `zone5r1c` is the one
custom solph component and the reference implementation for writing your
own.
"""

from . import stock, zone5r1c  # noqa: F401  (imported for registration)
from .zone5r1c import ThermalZone5R1C, ThermalZone5R1CBlock

__all__ = ["ThermalZone5R1C", "ThermalZone5R1CBlock"]
