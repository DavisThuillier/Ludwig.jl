###
### Diagonalize the IBZ-restricted electron-electron collision operator
###
### Reconstructs the full-BZ operator from the IBZ-restricted collision matrix `L` (N × N_ibz),
### diagonalizes it, and writes the eigenvalues, eigenvectors, full-BZ energies, and detailed-balance
### factors `D` back into the same HDF5 file as new datasets alongside "L". With --force, existing
### results are overwritten.
###
### The grid, lattice/point-group data, and build temperature come from the self-contained collision
### file; the material file supplies the dispersion.
###
### Run (serial):
###     julia diagonalize_collision_operator.jl <collision_file> <material_file>
###

using Pkg
Pkg.activate(@__DIR__)

using ArgParse
using HDF5
using StaticArrays
using Ludwig
using LinearAlgebra

function argument_handling(ARGS)
    settings = ArgParseSettings(description = "Diagonalize the IBZ-restricted electron-electron collision operator.")

    @add_arg_table! settings begin
        "--force", "-f"
            help = "Re-diagonalize and overwrite existing eigenvalue/eigenvector datasets"
            action = :store_true
        "collision_file"
            help = "Self-contained collision HDF5 file from electron_electron.jl; also the write target"
            arg_type = String
            required = true
        "material_file"
            help = "File defining the dispersion (and the rest of the model)"
            arg_type = String
            required = true
    end

    return parse_args(ARGS, settings)
end

"""
    CollisionData

The contents of a self-contained collision file: the IBZ matrix `L`, the full-BZ quadrature grid
(`nodes`, `weights`, per-node `energies`), the detailed-balance `balance_factors`, the lattice
`primitives` and Cartesian point-group `generators`, and the provenance scalars (`T_kelvin`, `N_ibz`,
`N`, `model_name`, …).

`energies` and `balance_factors` are `nothing` for older files that predate those datasets; callers
fall back to evaluating the material dispersion / recomputing the factors in that case.
"""
struct CollisionData{D}
    L::Matrix{Float64}
    nodes::Vector{SVector{D,Float64}}
    weights::Vector{Float64}
    energies::Union{Vector{Float64},Nothing}
    balance_factors::Union{Vector{Float64},Nothing}
    primitives::SMatrix{D,D,Float64}
    generators::Vector{SMatrix{D,D,Float64}}
    provenance::Dict{String,Any}
end

"""
    read_collision_operator(path) -> CollisionData

Read the matrix, grid, lattice/point-group data, and provenance attributes from a self-contained
collision file written by `electron_electron.jl`. The per-node `energies` and detailed-balance `"D"`
datasets are read when present and left as `nothing` for older files that lack them.
"""
function read_collision_operator(path)
    L, node_matrix, weights, energies, balance_factors, primitive_matrix, generator_stack, provenance = h5open(path, "r") do file
        provenance = Dict{String,Any}()
        for key ∈ keys(attributes(file["provenance"]))
            provenance[key] = read_attribute(file["provenance"], key)
        end
        energies = haskey(file, "energies") ? Vector{Float64}(read(file, "energies")) : nothing
        balance_factors = haskey(file, "D") ? Vector{Float64}(read(file, "D")) : nothing
        (
            Matrix{Float64}(read(file, "L")),
            Matrix{Float64}(read(file, "nodes")),
            Vector{Float64}(read(file, "weights")),
            energies,
            balance_factors,
            Matrix{Float64}(read(file, "lattice/primitives")),
            Array{Float64,3}(read(file, "lattice/generators")),
            provenance,
        )
    end

    D = size(node_matrix, 1)
    nodes      = [SVector{D,Float64}(view(node_matrix, :, i)) for i ∈ axes(node_matrix, 2)]
    primitives = SMatrix{D,D,Float64}(primitive_matrix)
    generators = [SMatrix{D,D,Float64}(view(generator_stack, :, :, g)) for g ∈ axes(generator_stack, 3)]

    return CollisionData{D}(L, nodes, weights, energies, balance_factors, primitives, generators, provenance)
end

function main(parsed)
    @info "starting diagonalize_collision_operator"; flush(stderr)

    collision_file = parsed["collision_file"]
    force          = parsed["force"]

    done = h5open(collision_file, "r") do file
        haskey(file, "eigenvalues") && haskey(file, "eigenvectors")
    end
    if done && !force
        @info "$collision_file is already diagonalized; pass --force to recompute"
        return nothing
    end

    data        = read_collision_operator(collision_file)
    L           = data.L
    nodes       = data.nodes
    weights     = data.weights
    T           = kb * Float64(data.provenance["T_kelvin"])   # energy units (eV)
    point_group = PointGroup(data.generators, data.primitives)

    # Per-node band energies: the stored dataset on current files, else the material dispersion for
    # older files that predate it.
    energies = data.energies === nothing ? dispersion.(nodes) : data.energies

    # Quadrature-weighted detailed-balance factors Dᵢ = wᵢ·nᵢ(1 - nᵢ), from the stored dataset on
    # current files, else recomputed for older files that predate it. The stored matrix L carries only
    # the column weight wⱼ, so the row weight wᵢ enters here for the symmetrized S = D^{1/2}·L·D^{-1/2}
    # to be symmetric.
    D = data.balance_factors !== nothing ? data.balance_factors :
        detailed_balance_factors(weights, energies, T)

    enforce_particle_conservation!(L, D)

    L_full = full_collision_matrix(L, point_group; nodes)
    eigenvalues, eigenvectors = collision_eigen(L_full, D)

    h5open(collision_file, "r+") do file
        for name ∈ ("eigenvalues", "eigenvectors", "energies", "D")
            haskey(file, name) && delete_object(file, name)
        end
        file["eigenvalues"]  = eigenvalues
        file["eigenvectors"] = eigenvectors
        file["energies"]     = energies
        file["D"]            = D         # Symmetrization factor
    end

    @info "wrote $(length(eigenvalues)) eigenvalues, $(size(eigenvectors, 1))×$(size(eigenvectors, 2)) eigenvectors, $(length(energies)) energies, and $(length(D)) detailed-balance factors to $collision_file"
    return nothing
end

const PARSED = argument_handling(ARGS)
include(joinpath(pwd(), PARSED["material_file"]))         # defines dispersion/velocity/primitives/generators
main(PARSED)
