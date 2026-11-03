"""Display-only geometry cleanup for publication-scale world maps.

The policy calculations continue to use the untouched GADM-derived geometry.
This module only prepares a visually stable copy for global map rendering:
small interior rings are filled, while outlines are restricted to components
large enough to remain legible at the final page scale.
"""
from __future__ import annotations

import geopandas as gpd
import shapely
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon


DISPLAY_CRS = 6933
MIN_VISIBLE_HOLE_KM2 = 1_000.0
MIN_OUTLINED_COMPONENT_KM2 = 2_500.0


def _polygon_parts(geometry) -> list[Polygon]:
    """Return polygon parts from polygonal or collection geometry."""
    if geometry is None or geometry.is_empty:
        return []
    valid = shapely.make_valid(geometry)
    parts: list[Polygon] = []
    stack = list(shapely.get_parts(valid))
    while stack:
        part = stack.pop()
        if isinstance(part, Polygon):
            parts.append(part)
        elif isinstance(part, (MultiPolygon, GeometryCollection)):
            stack.extend(shapely.get_parts(part))
    return parts


def _assemble(parts: list[Polygon]):
    if not parts:
        return GeometryCollection()
    if len(parts) == 1:
        return parts[0]
    return MultiPolygon(parts)


def _clean_fill(geometry, min_hole_area_m2: float):
    cleaned: list[Polygon] = []
    for polygon in _polygon_parts(geometry):
        retained_holes = [
            ring.coords
            for ring in polygon.interiors
            if Polygon(ring).area >= min_hole_area_m2
        ]
        cleaned.append(Polygon(polygon.exterior.coords, retained_holes))
    return _assemble(cleaned)


def _clean_outline(geometry, min_component_area_m2: float):
    parts = _polygon_parts(geometry)
    if not parts:
        return GeometryCollection()
    retained = [part for part in parts if part.area >= min_component_area_m2]
    if not retained:
        retained = [max(parts, key=lambda part: part.area)]
    return _assemble(retained)


def prepare_display_regions(
    regions: gpd.GeoDataFrame,
    hole_threshold_km2: float = MIN_VISIBLE_HOLE_KM2,
    outline_threshold_km2: float = MIN_OUTLINED_COMPONENT_KM2,
) -> gpd.GeoDataFrame:
    """Attach cleaned fill and outline geometries without changing source data."""
    if regions.crs is None:
        raise ValueError("Display geometry requires an explicit source CRS")
    projected = regions.to_crs(DISPLAY_CRS).copy()
    min_hole_area_m2 = float(hole_threshold_km2) * 1_000_000.0
    min_component_area_m2 = float(outline_threshold_km2) * 1_000_000.0

    fill_projected = [
        _clean_fill(geometry, min_hole_area_m2)
        for geometry in projected.geometry
    ]
    outline_projected = [
        _clean_outline(geometry, min_component_area_m2)
        for geometry in fill_projected
    ]

    result = regions.copy()
    result["display_fill"] = gpd.GeoSeries(
        fill_projected, crs=DISPLAY_CRS
    ).to_crs(regions.crs).to_numpy()
    result["display_outline"] = gpd.GeoSeries(
        outline_projected, crs=DISPLAY_CRS
    ).to_crs(regions.crs).to_numpy()
    if result[["display_fill", "display_outline"]].isna().any().any():
        raise RuntimeError("Display geometry cleanup produced missing geometries")
    return result


def display_geometry_audit(regions: gpd.GeoDataFrame) -> dict[str, int]:
    """Summarize the number of rendered paths before and after cleanup."""
    original_parts = sum(len(_polygon_parts(g)) for g in regions.geometry)
    original_holes = sum(
        len(part.interiors)
        for geometry in regions.geometry
        for part in _polygon_parts(geometry)
    )
    outlined_parts = sum(
        len(_polygon_parts(g)) for g in regions["display_outline"]
    )
    retained_holes = sum(
        len(part.interiors)
        for geometry in regions["display_fill"]
        for part in _polygon_parts(geometry)
    )
    return {
        "original_parts": int(original_parts),
        "original_holes": int(original_holes),
        "outlined_parts": int(outlined_parts),
        "retained_holes": int(retained_holes),
    }
