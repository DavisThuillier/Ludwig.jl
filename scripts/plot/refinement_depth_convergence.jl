using Pkg
Pkg.activate(@__DIR__)

using Ludwig
using ProgressMeter
using CairoMakie
using Random

include(joinpath(@__DIR__, "..", "materials", "graphene.jl"))

function get_quadrature_rule(crystal, energy_window, N_f::Int, N_ε::Int; refinement_depth, progress = true)
    nodes, weights, residuals = Ludwig.bz_quadrature_rule(dispersion, velocity, crystal, energy_window, N_f, N_ε; initial_depth = 11, refinement_depth, progress)
    @show maximum(residuals; init = 0.0)

    return nodes, weights
end

"""
    sample_pairs(N, n_samples, rng) -> Vector{Tuple{Int,Int}}

Returns `n_samples` distinct random index pairs `(i, j)` with `i ≠ j`, each drawn from `1:N`.
"""
function sample_pairs(N::Int, n_samples::Int, rng::AbstractRNG)
    @assert N ≥ 2 "Need at least two nodes to sample i ≠ j pairs"
    n_max = N * (N - 1)
    n_samples = min(n_samples, n_max)

    pairs = Set{Tuple{Int,Int}}()
    while length(pairs) < n_samples
        i = rand(rng, 1:N)
        j = rand(rng, 1:N)
        i == j && continue
        push!(pairs, (i, j))
    end

    return collect(pairs)
end

function main()
    T = 0.0001 # In energy units
    energy_window = 12.0 * T
    N_s = 10
    N_ε = 12

    n_samples = 20
    initial_depth = 6
    refinement_depths = 14:2:22   # total marching depths whose convergence we track
    asymptotic_depth = 23        # reference depth treated as ground truth
    rng = MersenneTwister(20260624)

    point_group = PointGroup(generators, primitives)
    crystal = Crystal(primitives, point_group)

    nodes, weights = get_quadrature_rule(crystal, energy_window, N_s, N_ε; refinement_depth = 20)
    N = length(nodes)
    @show N

    pairs = sample_pairs(N, n_samples, rng)
    @show length(pairs)

    ###
    ### Compute relative-error histories
    ###
    # relative_errors[s, r] = |L(refinement_depths[r]) - L(asymptotic)| / |L(asymptotic)| for sample s
    relative_errors = fill(NaN, length(pairs), length(refinement_depths))

    progress = Progress(length(pairs) * length(refinement_depths); desc = "Sampling pairs: ")
    for (s, (i, j)) ∈ enumerate(pairs)
        ki = nodes[i]
        kj = nodes[j]
        wj = weights[j]

        L_asymptotic = electron_electron(ki, kj, wj, crystal, dispersion, velocity, T, Weff_squared; threaded = true, initial_depth, refinement_depth = asymptotic_depth)

        for (r, refinement_depth) ∈ enumerate(refinement_depths)
            Lij = electron_electron(ki, kj, wj, crystal, dispersion, velocity, T, Weff_squared; threaded = true, initial_depth, refinement_depth)
            next!(progress)
            relative_errors[s, r] = abs(Lij - L_asymptotic) / abs(L_asymptotic)
        end

    end
    finish!(progress)

    ###
    ### Plot convergence histories
    ###
    fig = Figure(size = (800, 600))
    ax = Axis(fig[1, 1];
        yscale = log10,
        xlabel = "refinement depth", ylabel = "relative error vs. depth = $(asymptotic_depth)",
        title = "Refinement-depth convergence of L_ij ($(length(pairs)) sampled pairs)")

    for s ∈ axes(relative_errors, 1)
        history = relative_errors[s, :]
        scatterlines!(ax, collect(refinement_depths), history; color = (:steelblue, 0.35), markersize = 6)
    end

    display(fig)

    output_path = joinpath(@__DIR__, "..", "plots", "refinement_depth_convergence_T-$(T).png")
    save(output_path, fig)
    @info "Saved convergence plot" output_path

    return relative_errors
end

main()
