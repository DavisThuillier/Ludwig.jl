# Marching Squares Algorithm

Ludwig generates the energy contours for its Fermi-surface centered meshes using a simple marching squares algorithm. 
This implementation is inspired by [Contour.jl](https://juliageometry.github.io/Contour.jl/stable/index.html), but has been modified to generate contours bounded by a convex polygon. This is useful for generating the contours only within the (irreducible) Brillouin zone to create meshes that fully preserve the symmetry of the lattice.

Resulting contours are returned in the convenient [`IsolineBundle`](@ref) which stores information about the size and topology of the Fermi surface.

## Convex-mask extension

A stock marching squares implementation produces contours over the entire sampled grid. For Ludwig's symmetry-preserving meshes we instead need contours that exist only inside the IBZ polygon and *terminate cleanly on its boundary*, since the boundary cuts of two symmetry-related patches must coincide for [`fill_from_ibz!`](@ref) to stitch them together without seams.

Clipping the output of an external contour library after the fact does not give this: a generated contour that exits and re-enters the IBZ would come back as several disconnected arcs with no record of which arcs originally belonged to the same contour, and the clip points would not be the same as the contour endpoints used during marching. To avoid this, the masking happens *during* contour generation.

When a convex `mask` is supplied, an internal cell-classification pass sorts every grid cell containing a level-set crossing into one of three categories by counting how many of the cell's four corners lie inside the polygon:

| corners inside | category | bookkeeping |
|----------------|----------|-------------|
| 0              | outside  | discarded |
| 1, 2, or 3     | border   | recorded as a border cell |
| 4              | inside   | recorded as an interior cell |

The contour-following routine then walks through both classes. In an interior cell the next crossing is just the linearly interpolated edge intersection. In a border cell the candidate next crossing is tested against every edge of the mask polygon; the first edge whose parametric intersection falls within the current segment is taken as the contour's terminating point, and the march for that contour stops there. The result is an [`Isoline`](@ref) whose endpoints lie exactly on the mask boundary — exactly what the patch-alignment step downstream requires.

## Contour-to-patch pipeline

The marching-squares output is the raw material for the Fermi-surface mesh. The full chain from a sampled energy grid to a list of [`Patch`](@ref)es is:

```
energy grid A(x, y)
        │
        │ contours(x, y, A, levels; mask)        ← marching squares + IBZ clip
        ▼
Vector{IsolineBundle}                             one bundle per energy level (ε-Δε, ε, ε+Δε)
        │
        │ match_contour_segments(b₋, b₀, b₊)     ← pair isolines across the three levels
        ▼
matched (ε-Δε, ε, ε+Δε) triples
        │
        │ find_cusp_fraction(central isoline)    ← detect van Hove cusps
        ▼
cusp arclength fractions
        │
        │ arclength_slice(central, n; pinned)    ← n equal-length cuts, cusps snapped
        │ aligned_arclength_slice(outer, …)      ← outer contours resampled to match
        ▼
corner points on three contours
        │
        │ pair corners across contours
        ▼
Vector{Patch} → assembled into a Mesh
```

Each step beyond the energy-grid sampling is an internal helper:

- `contours` is the top-level entry point to the marching squares routine described above; it returns one [`IsolineBundle`](@ref) per requested level, optionally clipped to a convex `mask`.
- `match_contour_segments` pairs the isolines of two bundles by centroid proximity, so that each contour at ``\varepsilon - \Delta\varepsilon`` and ``\varepsilon + \Delta\varepsilon`` is associated with its counterpart on the central contour at ``\varepsilon``.
- `find_cusp_fraction` walks the central contour and flags interior points where the turning angle exceeds a threshold (default ``\pi/8``), returning their arclength fractions in ``[0, 1]``. These mark van Hove cusps that should land exactly on a patch corner rather than be smoothed over by uniform sampling.
- `arclength_slice` cuts a contour into `n` points uniformly spaced in arclength, optionally snapping selected interior corners onto pinned arclengths supplied by the cusp detector.
- `aligned_arclength_slice` is the same routine applied to one of the outer contours, but with the traversal direction chosen so its first tangent agrees with the central contour's first tangent — this is what keeps a patch's three corners on the same side of the Fermi surface.
