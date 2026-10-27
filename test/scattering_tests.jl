using LinearAlgebra
using StaticArrays

###
### Shared workload
###

const T_scat     = 0.005
const Δε_scat    = 1.5 * T_scat
const n_arc_scat = 6
const N_grid_scat = 401
const lat_scat   = Lattice([1.0 0.0; 0.0 1.0])
const rlv_scat   = reciprocal_lattice_vectors(lat_scat)
const bz_scat    = get_bz(lat_scat)
ε_scat(k)        = -2.0 * (cos(k[1]) + cos(k[2]))

const mesh_scat = ibz_mesh(lat_scat, [ε_scat], T_scat, Δε_scat, n_arc_scat, N_grid_scat)
const grid_scat = patches(mesh_scat)
const f0s_scat  = f0.(energy.(grid_scat), T_scat)

Weff_squared(p1, p2, p3, p4; kwargs...) = 1.0

###
### Index helpers
###

function find_same_energy_pair(grid)
    for i in eachindex(grid), j in eachindex(grid)
        i == j && continue
        abs(grid[i].e - grid[j].e) < grid[i].de / 2 && return (i, j)
    end
    return nothing
end

function find_distant_energy_pair(grid)
    for i in eachindex(grid), j in eachindex(grid)
        i == j && continue
        abs(grid[i].e - grid[j].e) > 2 * grid[i].de && return (i, j)
    end
    return nothing
end

@testset "electron_electron: Lattice dispatch" begin
    L11 = electron_electron(
        grid_scat, f0s_scat, 1, 1, [ε_scat], T_scat, Weff_squared, rlv_scat, bz_scat
    )
    @test isa(L11, Float64)
    @test isfinite(L11)

    L12 = electron_electron(
        grid_scat, f0s_scat, 1, 2, [ε_scat], T_scat, Weff_squared, rlv_scat, bz_scat
    )
    @test isa(L12, Float64)
    @test isfinite(L12)

    N    = length(grid_scat)
    mid  = (N + 1) ÷ 2
    Lmid = electron_electron(
        grid_scat, f0s_scat, mid, mid, [ε_scat], T_scat, Weff_squared, rlv_scat, bz_scat
    )
    @test isfinite(Lmid)

    Lend = electron_electron(
        grid_scat, f0s_scat, N, 1, [ε_scat], T_scat, Weff_squared, rlv_scat, bz_scat
    )
    @test isfinite(Lend)
end

@testset "electron_electron: NoLattice dispatch" begin
    ε_iso(r) = 0.5 * (r^2 - 1.0)
    mesh_iso = isotropic_mesh(ε_iso, [1.0], T_scat, Δε_scat, n_arc_scat)
    grid_iso = patches(mesh_iso)
    f0s_iso  = f0.(energy.(grid_iso), T_scat)
    nl       = NoLattice()

    L11 = electron_electron(
        grid_iso, f0s_iso, 1, 1, ε_iso, T_scat, Weff_squared, nl
    )
    @test isa(L11, Float64)
    @test isfinite(L11)

    L12 = electron_electron(
        grid_iso, f0s_iso, 1, 2, ε_iso, T_scat, Weff_squared, nl
    )
    @test isfinite(L12)

    N = length(grid_iso)
    Lend = electron_electron(
        grid_iso, f0s_iso, N, 1, ε_iso, T_scat, Weff_squared, nl
    )
    @test isfinite(Lend)
end

###
### Vertex from Bloch states
###

# Two-orbital Bloch Hamiltonian on the square lattice whose lower band crosses zero energy
function H_states(k)
    s1, s2 = sin(k[1]), sin(k[2])
    d = 2.5 + cos(k[1]) + cos(k[2])
    return @SMatrix [2.5 + d s1 - im*s2; s1 + im*s2 2.5 - d]
end

lower_eigenvector(k) = eigen(Hermitian(H_states(k))).vectors[:, 1]

# The same screened-Coulomb vertex with orbital form factors, from patches and their states
# or from patches alone by diagonalizing at each one
function W_states(p1, p2, p3, p4, ψ1, ψ2, ψ3, ψ4)
    q = momentum(p1) - momentum(p3)
    return abs2(dot(ψ3, ψ1)) * abs2(dot(ψ4, ψ2)) / (dot(q, q) + 0.25)^2
end
W_patches(ps...) = W_states(ps..., lower_eigenvector.(momentum.(ps))...)

W_const(args...) = 1.0

@testset "electron_electron: states" begin
    bands = (HamiltonianBand(H_states, 1),)
    mesh  = ibz_mesh(lat_scat, collect(bands), T_scat, Δε_scat, n_arc_scat, N_grid_scat)
    grid  = patches(mesh)
    f0s   = f0.(energy.(grid), T_scat)
    N     = length(grid)

    ψs = eigenstates(grid, bands)
    @test length(ψs) == N
    for (p, ψ) in zip(grid, ψs)
        @test H_states(momentum(p)) * ψ ≈ bands[1](momentum(p)) * ψ
    end

    for i in 1:3
        L_states  = [electron_electron(grid, f0s, i, j, bands, T_scat, W_states,
                                       rlv_scat, bz_scat; states=ψs) for j in 1:N]
        L_patches = [electron_electron(grid, f0s, i, j, bands, T_scat, W_patches,
                                       rlv_scat, bz_scat) for j in 1:N]
        @test any(!iszero, L_patches)
        @test isapprox(L_states, L_patches; rtol=1e-10)

        # A vertex that ignores its arguments gives the same row on either path
        @test [electron_electron(grid, f0s, i, j, bands, T_scat, W_const,
                                 rlv_scat, bz_scat; states=ψs) for j in 1:N] ==
              [electron_electron(grid, f0s, i, j, bands, T_scat, W_const,
                                 rlv_scat, bz_scat) for j in 1:N]
    end

    @test_throws DimensionMismatch electron_electron(
        grid, f0s, 1, 1, bands, T_scat, W_states, rlv_scat, bz_scat; states=ψs[1:end-1]
    )
end

@testset "electron_impurity: Function dispatch" begin
    V_squared_fn(ka, kb) = 1.0

    @test electron_impurity(grid_scat, 1, 1, V_squared_fn) == 0.0

    same = find_same_energy_pair(grid_scat)
    @test same !== nothing
    i, j = same
    Lij = electron_impurity(grid_scat, i, j, V_squared_fn)
    @test isa(Lij, Float64)
    @test isfinite(Lij)
    @test Lij ≤ 0.0

    distant = find_distant_energy_pair(grid_scat)
    @test distant !== nothing
    di, dj = distant
    @test electron_impurity(grid_scat, di, dj, V_squared_fn) == 0.0
end

@testset "electron_impurity: Matrix dispatch" begin
    V_squared_mat = ones(1, 1)

    @test electron_impurity(grid_scat, 1, 1, V_squared_mat) == 0.0

    same = find_same_energy_pair(grid_scat)
    @test same !== nothing
    i, j = same
    Lij = electron_impurity(grid_scat, i, j, V_squared_mat)
    @test isa(Lij, Float64)
    @test isfinite(Lij)
    @test Lij ≤ 0.0

    distant = find_distant_energy_pair(grid_scat)
    @test distant !== nothing
    di, dj = distant
    @test electron_impurity(grid_scat, di, dj, V_squared_mat) == 0.0
end

@testset "electron_phonon: Lattice dispatch" begin
    g_coupling(ka, kb, q, ω0; kwargs...) = 1.0 + 0im
    ω_phonon(q) = 0.05 * norm(q)
    f0s_scat = f0.(energy.(grid_scat), T_scat)

    # Same-energy patches: kernel returns 0 by the energy-window guard at the
    # top of the function (phonons can only mediate inelastic transitions).
    same = find_same_energy_pair(grid_scat)
    @test same !== nothing
    i, j = same
    Lij_same = electron_phonon(grid_scat, f0s_scat, i, j, T_scat, g_coupling, ω_phonon, lat_scat)
    @test Lij_same == 0.0

    # Distant-energy patches: kernel returns a finite value.
    distant = find_distant_energy_pair(grid_scat)
    @test distant !== nothing
    di, dj = distant
    Lij_distant = electron_phonon(
        grid_scat, f0s_scat, di, dj, T_scat, g_coupling, ω_phonon, lat_scat
    )
    @test isa(Lij_distant, Float64)
    @test isfinite(Lij_distant)
end
