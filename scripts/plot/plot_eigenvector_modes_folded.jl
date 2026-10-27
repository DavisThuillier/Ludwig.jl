###
### Plot the slowest eigenvector modes of the collision operator over the folded Fermi pockets
###
### Every quadrature node is mapped into the fundamental cell of the reciprocal lattice, stitching the
### Brillouin-zone corner wedges into whole Fermi pockets, and each pocket is drawn in its own panel on a
### shared colour scale.
###

using Pkg
Pkg.activate(@__DIR__)

using CairoMakie

include(joinpath(@__DIR__, "local_settings.jl"))
const SETTINGS = LOCAL_SETTINGS.plot_eigenvector_modes_folded

# Supplies dispersion(k) for masking nodes outside the thermal energy window.
include(joinpath(@__DIR__, "..", "materials", SETTINGS.material_file))

include(joinpath(@__DIR__, "data_io.jl"))
include(joinpath(@__DIR__, "folded_pockets.jl"))

"""
    plot_eigenvector_modes_folded(path)

Display the eigenvectors of the `n_modes` smallest eigenvalues as smooth 2D heatmaps over the folded
Fermi pockets, one panel per pocket on a shared colour scale. Each panel is a Gouraud-shaded mesh over
the Delaunay triangulation of that pocket's folded nodes, overlaid with a grey mask covering everything
outside the thermal energy window. Display-only; nothing is written to disk. Warns and skips for non-2D
grids.
"""
function plot_eigenvector_modes_folded(path)
    data = load_eigenmodes(path)

    eigenvalues = data.eigenvalues

    N_plot = SETTINGS.n_modes
    n_available = length(eigenvalues)
    if n_available < N_plot
        @warn "fewer than $N_plot modes available ($n_available); skipping eigenvector heatmaps"
        return nothing
    end

    slowest = sortperm(eigenvalues)[1:N_plot]

    plot_folded_modes(eigenvalues[slowest], data.eigenvectors[:, slowest], data.nodes,
        data.lattice.primitives, data.provenance.energy_window, dispersion;
        subtitle = "$(data.provenance.model_name), T = $(data.provenance.T_kelvin) K",
        window_pad = SETTINGS.window_pad,
        show_seams = SETTINGS.show_seams,
    )

    return nothing
end

function main()
    collision_file = joinpath(@__DIR__, "..", "data", SETTINGS.collision_file)

    plot_eigenvector_modes_folded(collision_file)

    return nothing
end

main()
