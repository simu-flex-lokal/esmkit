# -*- coding: utf-8 -*-
"""
The framework layer: spec schema, component registry, builder, solver and
result extraction.

**This package must never import from `esmkit.components`.** Components are
plugins which register themselves; the core only ever knows the registry
mechanism. `test/test_seam.py` enforces it.
"""
