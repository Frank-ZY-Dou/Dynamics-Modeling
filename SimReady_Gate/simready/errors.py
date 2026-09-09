"""Errors raised by queries that could not be evaluated.

A ray cast or point-in-mesh test that fails (a missing backend, a degenerate mesh) tells the
caller nothing about the geometry, so it is raised and reported instead of being read as "no
hit" or "not contained".
"""


class GeometryQueryError(RuntimeError):
    """A required geometric query (ray cast, point containment) could not be evaluated."""
