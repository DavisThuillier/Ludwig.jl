###
### Diagnostic: is the balanced collision operator A = Diagonal(D)*L_full symmetric?
###
### Throwaway analysis script. Reconstructs L_full and D exactly as collision_eigen does (before the
### Symmetric() wrap) from the stored monolayer-graphene T=40 K collision file, then measures the
### asymmetry of A = Diagonal(D)*L_full against the magnitude of the negative eigenvalues.
###

using Pkg
Pkg.activate(@__DIR__)

using HDF5
using StaticArrays
using LinearAlgebra
using Ludwig

include(joinpath(@__DIR__, "local_settings.jl"))
const SETTINGS = LOCAL_SETTINGS.diagnose_collision_symmetry

function load()
    h5open(joinpath(@__DIR__, "..", "data", SETTINGS.collision_file), "r") do f
        L          = Matrix{Float64}(read(f, "L"))
        weights    = Vector{Float64}(read(f, "weights"))
        energies   = Vector{Float64}(read(f, "energies"))
        D_stored   = Vector{Float64}(read(f, "D"))
        node_mat   = Matrix{Float64}(read(f, "nodes"))
        prim_mat   = Matrix{Float64}(read(f, "lattice/primitives"))
        gen_stack  = Array{Float64,3}(read(f, "lattice/generators"))
        T_kelvin   = read_attribute(f["provenance"], "T_kelvin")
        Dm = size(node_mat, 1)
        nodes = [SVector{Dm,Float64}(view(node_mat, :, i)) for i in axes(node_mat, 2)]
        primitives = SMatrix{Dm,Dm,Float64}(prim_mat)
        generators = [SMatrix{Dm,Dm,Float64}(view(gen_stack, :, :, g)) for g in axes(gen_stack, 3)]
        return L, nodes, weights, energies, D_stored, primitives, generators, T_kelvin
    end
end

function main()
    L, nodes, weights, energies, D_stored, primitives, generators, T_kelvin = load()
    T = kb * Float64(T_kelvin)
    point_group = PointGroup(generators, primitives)

    # Reconstruct exactly as collision_eigen / diagonalize_collision_operator do (pre-Symmetric).
    D = detailed_balance_factors(weights, energies, T)
    L_full = full_collision_matrix(L, point_group; nodes)
    
    # Sanity: the D we recompute should match the stored D the diagonalizer used.
    println("max |D_recomputed - D_stored| / max|D| = ",
            maximum(abs, D .- D_stored) / maximum(abs, D))

    N = size(L_full, 1)
    println("N = ", N, "   T = ", T_kelvin, " K")

    ###
    ### Balanced operator A = Diagonal(D) * L_full  and its asymmetry
    ###
    # Ludwig.symmetrize_collision_matrix!(L_full, D)
    # enforce_particle_conservation!(L_full, D)
    A = Diagonal(D) * L_full
    rel_asym = norm(A - A') / norm(A)
    println("\n--- balanced operator A = Diagonal(D) * L_full ---")
    println("norm(A - A') / norm(A) = ", rel_asym)

    # return nothing

    # Localize the worst detailed-balance violation: D_i L[i,j] - D_j L[j,i].
    maxL = maximum(abs, L_full)
    worst = 0.0
    arg_i = 0; arg_j = 0
    @inbounds for j in 1:N, i in 1:N
        d = abs(A[i, j] - A[j, i])
        if d > worst
            worst = d; arg_i = i; arg_j = j
        end
    end
    println("max |D_i L[i,j] - D_j L[j,i]| / max|L_full| = ", worst / maxL,
            "   at (i,j) = (", arg_i, ",", arg_j, ")")

    @show A[arg_i, arg_j], A[arg_j, arg_i]

    ###
    ### Spectrum of the symmetrized operator (what collision_eigen actually diagonalizes)
    ###
    sqrtD = sqrt.(D)
    S = Symmetric(sqrtD .* L_full ./ transpose(sqrtD))
    λ = eigvals(S)
    λ_min = minimum(λ)
    λ_max = maximum(λ)
    ratio = abs(λ_min) / (λ_max - λ_min)
    nneg = count(<(0), λ)
    println("\n--- spectrum of S = D^{1/2} L_full D^{-1/2}  (Symmetric wrap) ---")
    println("λ_min (most negative) = ", λ_min)
    println("λ_max                 = ", λ_max)
    println("|λ_min| / (λ_max - λ_min) = ", ratio)
    println("number of negative eigenvalues = ", nneg, " / ", N)

    # Also the spectrum of the genuinely-asymmetric A itself (general eigensolver), for comparison.
    λA = eigvals(A)
    println("\n--- spectrum of A directly (general, may be complex) ---")
    println("max |imag(λ_A)| / max|real(λ_A)| = ",
            maximum(abs, imag.(λA)) / maximum(abs, real.(λA)))
    realλA = real.(λA)
    println("min real(λ_A) = ", minimum(realλA), "   max real(λ_A) = ", maximum(realλA))

    ###
    ### Verdict numbers side by side
    ###
    println("\n=== comparison ===")
    println("relative asymmetry      norm(A-A')/norm(A)         = ", rel_asym)
    println("relative negative depth |λ_min|/(λ_max - λ_min)    = ", ratio)
    println("ratio of the two (asym / negdepth)                 = ", rel_asym / ratio)
end

main()
