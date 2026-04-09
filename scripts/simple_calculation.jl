using Ludwig
using LinearAlgebra

function (@main)(_)
    ## Model ##############################################################

    # Square lattice with unit lattice constant
    l = Lattice([1.0 0.0; 0.0 1.0])

    # NNTB dispersion (energies in eV, momenta in units of 2π/a)
    t = 1.0   # eV
    μ = -1.0  # eV
    ε(k) = -2t * (cos(2π * k[1]) + cos(2π * k[2])) - μ

    # Temperature (in eV, same units as the Hamiltonian)
    T = 0.025  # ≈ 290 K

    ## BZ mesh ############################################################

    n_levels = 4   # number of energy contour boundaries (n_levels - 1 patches in energy)
    n_cuts   = 10  # number of angular cuts along the Fermi surface (n_cuts - 1 patches per sector)

    bz   = bz_mesh(l, [ε], T, n_levels, n_cuts)
    grid = bz.patches
    N    = length(grid)

    println("Total BZ patches: ", N)

    ## Symmetry map #######################################################

    sym = bz_symmetry_map(grid, l)
    println("IBZ patches:      ", length(sym.ibz_inds))

    ## Scattering vertex ##################################################

    U = 1.0  # eV, Hubbard interaction strength
    Weff_squared(p1, p2, p3, p4; kwargs...) = U^2

    ## Collision matrix ###################################################

    f0s = f0.(energy.(grid), T)

    println("Building $(N)×$(N) collision matrix (IBZ rows only)...")
    L = zeros(N, N)
    Threads.@threads for i in sym.ibz_inds
        for j in 1:N
            L[i, j] = electron_electron(grid, f0s, i, j, [ε], T, Weff_squared, l)
        end
    end
    fill_from_ibz!(L, sym)

    # Impose particle conservation: set diagonal to negative row sum
    for i in 1:N
        L[i, i] -= sum(L[i, :])
    end

    # Particle conservation: L * 1 ≈ 0
    residual = maximum(abs.(L * ones(N))) / maximum(abs.(diag(L)))
    @assert residual < 0.05 "Particle conservation violated: relative residual = $residual"
    println("Particle conservation residual: ", residual, "  ✓")

    ## Symmetrize #########################################################

    dV = [p.dV for p in grid]
    E  = energy.(grid)
    L  = symmetrize(L, dV, E, T)

    # L^(s) should be symmetric
    Ls = diagm(f0s .* (1 .- f0s) .* dV) * L
    @assert isapprox(Ls, Ls'; rtol = 1e-6) "Symmetrized auxiliary matrix L^(s) is not symmetric!"
    println("L^(s) symmetry check:          ✓")

    ## Conductivity #######################################################

    v = velocity.(grid)
    σ = Ludwig.electrical_conductivity(L, v, E, dV, T)
    println("\nσ_xx = ", real(σ[1, 1]))
    println("σ_xy = ", real(σ[1, 2]))

    # σ_xx should be positive; σ_xy should vanish by square symmetry
    @assert real(σ[1, 1]) > 0 "σ_xx is not positive!"
    @assert abs(real(σ[1, 2])) / real(σ[1, 1]) < 1e-6 "σ_xy is unexpectedly large!"
    println("Sign and symmetry of σ:        ✓")
end
