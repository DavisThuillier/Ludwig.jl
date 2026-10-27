###
### Plot a single row L_{i,j} of the electron-electron collision operator over the Fermi annulus
###
### Reads an HDF5 collision file, reconstructs the full-BZ operator from its IBZ block, fixes k_i
### to a chosen row index, and renders L_{i,j} as a tricontourf heatmap over the Fermi annulus.
###

using Pkg
Pkg.activate(@__DIR__)

using CairoMakie
using StaticArrays
using Ludwig

include(joinpath(@__DIR__, "local_settings.jl"))
const SETTINGS = LOCAL_SETTINGS.plot_collision_row

# Supplies dispersion(k) for masking nodes outside the thermal energy window.
include(joinpath(@__DIR__, "..", "materials", SETTINGS.material_file))

include(joinpath(@__DIR__, "data_io.jl"))

"""
    plot_collision_row(path, i_ref)

Display row `i_ref` of the full-BZ collision operator as a smooth 2D heatmap over k-space. The IBZ block
is unfolded to the full operator by point-group invariance, the fixed node k_i is marked with a red star,
and everything outside the thermal energy window or the Brillouin zone is masked in grey. Display-only;
nothing is written to disk. Warns and skips for non-2D grids.
"""
function plot_collision_row(path, i_ref)
    data = load_collision_operator(path)

    D = length(eltype(data.nodes))
    if D != 2
        @warn "collision-row heatmaps are only rendered for 2D grids; skipping (D = $D)"
        return nothing
    end

    nodes = data.nodes
    kx = first.(nodes)
    ky = last.(nodes)

    # Brillouin zone in the same Cartesian k-frame as the nodes; the file's primitives are authoritative.
    rlv = 2π * transpose(inv(data.lattice.primitives))
    bz = brillouin_zone(rlv)

    # Ordered closed boundary of the convex BZ polygon: corners sorted by polar angle about
    # their centroid, with the first vertex appended to close the loop.
    corners = vertices(bz)
    centroid = sum(corners) / length(corners)
    sort!(corners; by = corner -> atan(corner[2] - centroid[2], corner[1] - centroid[1]))
    boundary = push!(copy(corners), corners[1])
    boundary_x = first.(boundary)
    boundary_y = last.(boundary)

    # Unfold the IBZ block to the full-BZ operator and pull out the requested row.
    point_group = PointGroup(data.lattice.generators, data.lattice.primitives)
    L_full = Ludwig.full_collision_matrix(data.L, point_group; nodes)
    row = L_full[i_ref, :]
    ki = nodes[i_ref]
    color_extent = maximum(abs, L_full)

    # A cell is coloured only if it lies within both the thermal energy window and the BZ.
    inside(k) = abs(dispersion(k)) <= data.provenance.energy_window &&
                Ludwig.in_convex_polytope(k, bz)

    # Dense grid spanning the node bounding box, padded so the BZ boundary stands out against grey.
    pad = 0.15 * max(maximum(kx) - minimum(kx), maximum(ky) - minimum(ky))
    mask_resolution = 5000
    mask_x = range(minimum(kx) - pad, maximum(kx) + pad; length = mask_resolution)
    mask_y = range(minimum(ky) - pad, maximum(ky) + pad; length = mask_resolution)

    fig = Figure(size = (600, 550))
    ax = Axis(fig[1, 1];
        xlabel = L"k_x",
        ylabel = L"k_y",
        title  = L"L_{i,j}",
        subtitle = "$(data.provenance.model_name), T = $(data.provenance.T_kelvin) K, k_i = ($(round(ki[1]; sigdigits = 4)), $(round(ki[2]; sigdigits = 4)))",
        aspect = DataAspect(),
    )
    SETTINGS.xlims === nothing || xlims!(ax, SETTINGS.xlims...)
    SETTINGS.ylims === nothing || ylims!(ax, SETTINGS.ylims...)

    # Symmetric levels about zero give a diverging field centered on L = 0; tricontourf takes its
    # colour extent from `levels`, not a `colorrange`.
    tricontourf!(ax, kx, ky, row;
        colormap = :balance, levels = range(-color_extent, color_extent; length = 21))

    mask = [inside(SVector(x, y)) ? NaN : 1.0 for x ∈ mask_x, y ∈ mask_y]
    heatmap!(ax, mask_x, mask_y, mask;
        colormap = [:grey],
        nan_color = (:white, 0),
        colorrange = (0, 1),
    )

    # BZ boundary on top of the field and mask.
    lines!(ax, boundary_x, boundary_y; color = :black, linewidth = 2)

    # Mark the fixed node k_i on top of everything.
    # scatter!(ax, [ki[1]], [ki[2]]; color = :red, markersize = 8, marker = :star5)

    display(fig)

    return nothing
end

function main()
    collision_file = joinpath(@__DIR__, "..", "data", SETTINGS.collision_file)

    for i_ref ∈ SETTINGS.row_indices
        plot_collision_row(collision_file, i_ref)
    end

    return nothing
end

main()
