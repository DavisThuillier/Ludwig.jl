###
### Node fields drawn over the folded Fermi pockets
###
### Every quadrature node is mapped into the fundamental cell of the reciprocal lattice, stitching the
### Brillouin-zone corner wedges into whole Fermi pockets, and each pocket is drawn in its own panel on
### a shared colour scale. Shared by the scripts that render a per-node field this way: the collision
### operator's eigenvector modes read from a file, and any field a script holds in memory.
###

using CairoMakie
using LinearAlgebra
using StaticArrays
using Ludwig

# Triangulator for the Gouraud-shaded fields, reached through Makie's transitively loaded copy (the
# same one its `tricontourf` uses) so no direct dependency is required.
const DelTri = CairoMakie.Makie.DelTri

###
### Folded geometry
###

# Fractional coordinates come from the exact identity inv(rlv) = transpose(primitives)/(2π), so the fold
# never inverts the reciprocal vectors and carries no lattice-specific constants.
"""
    fold(k, rlv, lattice_primitives) -> SVector

Cartesian `k` mapped into the fundamental cell of the reciprocal lattice spanned by `rlv`.
"""
fold(k, rlv, lattice_primitives) = k - rlv * round.(transpose(lattice_primitives) * k / (2π))

"""
    pocket_centres(corners, rlv, lattice_primitives) -> Vector{SVector{2,Float64}}

The distinct folded images of the Brillouin-zone `corners`, ordered by polar angle. Fermi pockets at the
zone corners collapse onto these points, one per inequivalent valley.
"""
function pocket_centres(corners, rlv, lattice_primitives)
    tolerance = 1e-8 * maximum(norm, eachcol(rlv))
    centres = SVector{2,Float64}[]
    for corner ∈ corners
        centre = fold(corner, rlv, lattice_primitives)
        any(existing -> norm(existing - centre) < tolerance, centres) || push!(centres, centre)
    end
    sort!(centres; by = centre -> atan(centre[2], centre[1]))
    return centres
end

"""
    nearest_pocket(k, centres) -> Int

Index of the pocket centre closest to `k`.
"""
nearest_pocket(k, centres) = argmin(i -> norm(k - centres[i]), eachindex(centres))

"""
    pocket_panels(nodes, corners, rlv, lattice_primitives, inside; window_pad, mask_resolution)
        -> (; panels, radius)

Per-pocket drawing geometry and the half-width of the square panel window every pocket shares. Each
panel carries its pocket's folded `node_points` and their triangulation `faces`, the `mask_x`/`mask_y`
grid and `mask` covering the cells `inside` rejects, the `shifts` that translate the zone boundary onto
the seams meeting at the centre, and the `color_indices` selecting this pocket's components of a node
field.
`panels` is empty when no node falls into a folded zone corner.
"""
function pocket_panels(nodes, corners, rlv, lattice_primitives, inside;
                       window_pad = 1.15, mask_resolution = 1000)
    # Folding collapses the zone-corner wedges onto one point per inequivalent valley; each node and each
    # corner then belongs to the pocket whose centre it lands nearest.
    folded_nodes = [fold(k, rlv, lattice_primitives) for k ∈ nodes]
    centres = pocket_centres(corners, rlv, lattice_primitives)
    node_pocket = [nearest_pocket(k, centres) for k ∈ folded_nodes]
    corner_pocket = [nearest_pocket(fold(c, rlv, lattice_primitives), centres) for c ∈ corners]

    pockets = [(centre = centres[p], indices = findall(==(p), node_pocket),
                shifts = [centres[p] - corners[j] for j ∈ findall(==(p), corner_pocket)])
               for p ∈ eachindex(centres)]
    filter!(pocket -> !isempty(pocket.indices), pockets)
    isempty(pockets) && return (; panels = [], radius = 0.0)

    # One square window per pocket, sized by the farthest node of any pocket so every panel shares a scale.
    radius = window_pad *
             maximum(norm(folded_nodes[i] - pocket.centre) for pocket ∈ pockets for i ∈ pocket.indices)

    panels = map(pockets) do pocket
        kx = [folded_nodes[i][1] for i ∈ pocket.indices]
        ky = [folded_nodes[i][2] for i ∈ pocket.indices]

        # Triangulating in folded coordinates is what stitches the wedges together; the hull of an annular
        # pocket fills its hole, and the energy mask removes those cells.
        triangulation = DelTri.triangulate(permutedims([kx ky]))
        faces = reduce(vcat, ([triangle[1] triangle[2] triangle[3]]
                              for triangle ∈ DelTri.each_solid_triangle(triangulation)))

        mask_x = range(pocket.centre[1] - radius, pocket.centre[1] + radius; length = mask_resolution)
        mask_y = range(pocket.centre[2] - radius, pocket.centre[2] + radius; length = mask_resolution)
        mask = [inside(SVector(x, y)) ? NaN : 1.0 for x ∈ mask_x, y ∈ mask_y]

        return (; pocket.centre, pocket.shifts, node_points = [kx ky], faces, mask_x, mask_y, mask,
                  color_indices = pocket.indices)
    end

    return (; panels, radius)
end

###
### Rendering
###

"""
    plot_folded_modes(eigenvalues, eigenvectors, nodes, lattice_primitives, energy_window, dispersion;
                      subtitle, window_pad, show_seams)

Display every column of `eigenvectors` as a smooth 2D heatmap over the folded Fermi pockets, one figure
per column and one panel per pocket on a shared colour scale, headed by the matching entry of
`eigenvalues` and by `subtitle`. Each panel is a Gouraud-shaded mesh over the Delaunay triangulation of
that pocket's folded `nodes`, overlaid with a grey mask covering everything outside the `dispersion`
window `energy_window`. Display-only; nothing is written to disk. Warns and skips for non-2D grids.
"""
function plot_folded_modes(eigenvalues, eigenvectors, nodes, lattice_primitives, energy_window,
                           dispersion; subtitle = "", window_pad = 1.15, show_seams = true)
    # A column is a field over the whole Brillouin zone, so an IBZ-length vector belongs to a different
    # grid than `nodes` and would otherwise fail deep inside the per-pocket triangulation.
    size(eigenvectors, 1) == length(nodes) || throw(DimensionMismatch(
        "eigenvectors have $(size(eigenvectors, 1)) components but nodes holds $(length(nodes)) entries"))
    size(eigenvectors, 2) == length(eigenvalues) || throw(DimensionMismatch(
        "eigenvectors has $(size(eigenvectors, 2)) columns but eigenvalues holds $(length(eigenvalues)) entries"))

    D = length(eltype(nodes))
    if D != 2
        @warn "eigenvector heatmaps are only rendered for 2D grids; skipping (D = $D)"
        return nothing
    end

    # Brillouin zone in the same Cartesian k-frame as the nodes; the caller's primitives are
    # authoritative, since a fold built from a mismatched material would silently misplace every node.
    rlv = 2π * transpose(inv(lattice_primitives))
    bz = brillouin_zone(rlv)

    # Ordered closed boundary of the convex BZ polygon: corners sorted by polar angle about
    # their centroid, with the first vertex appended to close the loop.
    corners = vertices(bz)
    centroid = sum(corners) / length(corners)
    sort!(corners; by = corner -> atan(corner[2] - centroid[2], corner[1] - centroid[1]))
    boundary = push!(copy(corners), corners[1])
    boundary_x = first.(boundary)
    boundary_y = last.(boundary)

    # A cell is coloured only if it lies within the thermal energy window. The dispersion is
    # reciprocal-lattice periodic, so this test survives the fold; a zone-membership test does not, since
    # a folded pocket straddles the zone boundary and would be cut back to a single wedge.
    inside(k) = abs(dispersion(k)) <= energy_window

    # Every mode reuses the same faces and mask and only swaps the per-vertex colour.
    panels, radius = pocket_panels(nodes, corners, rlv, lattice_primitives, inside; window_pad)
    if isempty(panels)
        @warn "no nodes fell into any folded zone corner; skipping eigenvector heatmaps"
        return nothing
    end

    for mode ∈ axes(eigenvectors, 2)
        ψ = eigenvectors[:, mode]
        color_extent = maximum(abs, ψ)

        # The heading spans every column explicitly and reports no width, so the panels keep equal
        # widths and stay the same size under DataAspect.
        fig = Figure(size = (500 * length(panels) + 140, 560))
        heading = 1:(length(panels) + 1)
        Label(fig[0, heading], L"\psi_{%$mode},\; \lambda = %$(round(eigenvalues[mode]; sigdigits = 4))";
            fontsize = 20, tellwidth = false)
        Label(fig[1, heading], subtitle; fontsize = 13, tellwidth = false)

        for (col, panel) ∈ enumerate(panels)
            ax = Axis(fig[2, col];
                xlabel = L"k_x",
                ylabel = L"k_y",
                title  = "pocket at ($(round(panel.centre[1]; digits = 3)), $(round(panel.centre[2]; digits = 3)))",
                aspect = DataAspect(),
            )
            xlims!(ax, panel.centre[1] - radius, panel.centre[1] + radius)
            ylims!(ax, panel.centre[2] - radius, panel.centre[2] + radius)

            # Gouraud-shaded per-vertex colour renders the diverging field as a continuous surface over the
            # node triangulation, with the colour extent symmetric about ψ = 0 for this mode.
            mesh!(ax, panel.node_points, panel.faces; color = ψ[panel.color_indices],
                colormap = :balance, colorrange = (-color_extent, color_extent), shading = NoShading)

            heatmap!(ax, panel.mask_x, panel.mask_y, panel.mask;
                colormap = [:grey],
                nan_color = (:white, 0),
                colorrange = (0, 1),
            )

            # One translated zone boundary per corner folding onto this centre: together they trace the
            # seams between the wedges the pocket is stitched from.
            if show_seams
                for shift ∈ panel.shifts
                    lines!(ax, boundary_x .+ shift[1], boundary_y .+ shift[2];
                        color = (:black, 0.35), linewidth = 1)
                end
            end
        end

        Colorbar(fig[2, length(panels) + 1];
            colormap = :balance, colorrange = (-color_extent, color_extent),
            label = L"\psi_{%$mode}")

        display(fig)
    end

    return nothing
end
