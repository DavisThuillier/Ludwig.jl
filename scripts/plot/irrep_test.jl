using Pkg
Pkg.activate(@__DIR__)

using Ludwig
using CairoMakie
using HDF5
using LinearAlgebra
using StaticArrays
using KrylovKit
using LaTeXStrings
using SparseArrays

include(joinpath(@__DIR__, "local_settings.jl"))
const SETTINGS = LOCAL_SETTINGS.irrep_test

include(joinpath(@__DIR__, "..", "materials", SETTINGS.material_file))

include(joinpath(@__DIR__, "data_io.jl"))
include(joinpath(@__DIR__, "folded_pockets.jl"))


"""
    degeneracy_residual(values, multiplicity) -> Float64

Largest spread within consecutive groups of `multiplicity` ascending `values`, relative to the
spectral radius. Schur's lemma makes every eigenvalue of the isotypic block of a `d`-dimensional
irreducible representation exactly `d`-fold degenerate, so this measures how far the block departs
from an exact compression.
"""
function degeneracy_residual(values, multiplicity)
    multiplicity == 1 && return 0.0
    spread = maximum(Iterators.partition(eachindex(values), multiplicity)) do group
        values[last(group)] - values[first(group)]
    end
    return spread / maximum(abs, values)
end

"""
    spectrum_residual(blocks, reference) -> Float64

Largest deviation of the pooled spectrum of every isotypic block from the `reference` spectrum,
relative to the spectral radius. The blocks partition the full-BZ operator, so the two agree to
rounding whenever the construction is sound.
"""
function spectrum_residual(blocks, reference)
    spectrum = sort!(reduce(vcat, eigvals(Ludwig.symmetric_part(block.matrix)) for block ∈ blocks))
    length(spectrum) == length(reference) || throw(DimensionMismatch(
        "the blocks hold $(length(spectrum)) eigenvalues but the reference spectrum has $(length(reference))"))
    return maximum(abs, spectrum - sort(reference)) / maximum(abs, reference)
end

###
### Grid assumptions the block construction rests on
###

"""
    assert_block_layout(nodes, point_group, N_ibz; atol)

Check that `nodes` holds `N_ibz` representatives followed by their images under the remaining
operations, positional block `g` carrying `operations(point_group)[g]`. Every index identity used to
unfold the stored collision block assumes this, and no downstream result can reveal its failure.
"""
function assert_block_layout(nodes, point_group, N_ibz; atol = 1e-8)
    length(nodes) == N_ibz * order(point_group) || throw(DimensionMismatch(
        "$(length(nodes)) nodes is not $(order(point_group)) blocks of $N_ibz representatives"))
    for (g, M) ∈ enumerate(operations(point_group)), r ∈ 1:N_ibz
        norm(nodes[(g - 1) * N_ibz + r] - M * nodes[r]) ≤ atol || throw(ArgumentError(
            "node $((g - 1) * N_ibz + r) is not `operations(point_group)[$g] · nodes[$r]`; the grid " *
            "was built from a different ordering of the material's generators, so every block " *
            "carries a different operation than assumed"))
    end
    return nothing
end

"""
    block_constancy(D, N_ibz) -> Float64

Largest relative deviation of the detailed-balance factors `D` from their values on the first block.
They are constant over each point-group orbit whenever the quadrature weights are replicated and the
dispersion is invariant, which is what lets the symmetrization commute with the change of basis.
"""
function block_constancy(D, N_ibz)
    return maximum(abs(D[i] - D[(i - 1) % N_ibz + 1]) for i ∈ eachindex(D)) / maximum(abs, D)
end

"""
    stored_eigenvalues(path) -> Union{Vector{Float64},Nothing}

Spectrum of the collision file at `path`, or `nothing` when it has not been through
diagonalize_collision_operator.jl.
"""
function stored_eigenvalues(path)
    h5open(path, "r") do file
        haskey(file, "eigenvalues") ? Vector{Float64}(read(file, "eigenvalues")) : nothing
    end
end

"""
    symmetrize_L(data) -> Matrix{Float64}

Detailed-balance form `D^{1/2}·L·D^{-1/2}` of the stored IBZ block. Its symmetric part carries the
physical spectrum; that symmetrization is deferred to the isotypic blocks, where it acts on far
smaller matrices.
"""
function symmetrize_L(data)
    L_sym = similar(data.L)
    sqrtD = sqrt.(data.D)
    for i ∈ axes(L_sym)[1], j ∈ axes(L_sym)[2]
        L_sym[i,j] = (sqrtD[i] / sqrtD[j]) * data.L[i,j]
    end
    return L_sym
end

function main()
    # Which irreducible representation to diagonalize, how many of its slowest modes to draw, and
    # whether to show the raw block heatmaps.
    irrep_to_diagonalize = "E2"
    n_modes = 4
    show_block_heatmaps = false

    collision_path = joinpath(@__DIR__, "..", "data", SETTINGS.collision_file)
    rule = load_quadrature_rule(joinpath(@__DIR__, "..", "meshes", SETTINGS.quad_file))
    N_ibz = rule.header.N_ibz
    point_group = PointGroup(generators, primitives)

    # One decomposition serves both the symbols and the blocks. Lifting the degeneracy between
    # isotypic components needs a random draw of class coefficients, so a second call orders the
    # representations differently and any index taken from this one would then name a different block.
    table = character_table(point_group)
    @show table

    data = load_collision_operator(collision_path)
    data.D === nothing && error("$(SETTINGS.collision_file) has no `D` dataset; rebuild it with a " *
                                "current electron_electron.jl, or run diagonalize_collision_operator.jl on it")
    N_ibz == data.provenance.N_ibz || error(
        "$(SETTINGS.quad_file) holds $N_ibz IBZ representatives but $(SETTINGS.collision_file) was " *
        "built on $(data.provenance.N_ibz); the rule must be the one the collision file came from")

    assert_block_layout(data.nodes, point_group, N_ibz)
    @info "detailed-balance factors vary over each point-group orbit by $(block_constancy(data.D, N_ibz))"

    enforce_particle_conservation!(data.L, data.D)
    L = symmetrize_L(data)
    blocks = isotypic_blocks(L, N_ibz, table)

    if show_block_heatmaps
        for block ∈ blocks
            fig = Figure()
            ax = Axis(fig[1,1];
                aspect = DataAspect()
            )
            heatmap!(ax, block.matrix; colormap = :lisbon,
                colorrange = (-1e-9, 1e-9)
            )
            display(fig)
        end
    end

    reference = stored_eigenvalues(collision_path)
    if reference === nothing
        @info "$(SETTINGS.collision_file) holds no spectrum to check the blocks against; " *
              "run diagonalize_collision_operator.jl on it to enable the cross-check"
    else
        @info "pooled block spectrum matches the stored spectrum to $(spectrum_residual(blocks, reference))"
    end

    block = blocks[irrep_to_diagonalize]
    values, vectors = block_modes(block, N_ibz, sqrt.(data.D))

    modes = 1:min(n_modes, length(values))
    plot_folded_modes(values[modes], vectors[:, modes], data.nodes, data.lattice.primitives,
                      data.provenance.energy_window, dispersion;
                      subtitle = "$(data.provenance.model_name), T = $(data.provenance.T_kelvin) K " *
                                 "— $irrep_to_diagonalize modes",
                      window_pad = SETTINGS.window_pad, show_seams = SETTINGS.show_seams)

    return nothing
end

main()
