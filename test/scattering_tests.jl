using LinearAlgebra

###
### Shared workload
###

const T_scat     = 0.005
const Δε_scat    = 1.5 * T_scat
const n_arc_scat = 6
const N_grid_scat = 401
const lat_scat   = Lattice([1.0 0.0; 0.0 1.0])
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
        grid_scat, f0s_scat, 1, 1, [ε_scat], T_scat, Weff_squared, lat_scat
    )
    @test isa(L11, Float64)
    @test isfinite(L11)

    L12 = electron_electron(
        grid_scat, f0s_scat, 1, 2, [ε_scat], T_scat, Weff_squared, lat_scat
    )
    @test isa(L12, Float64)
    @test isfinite(L12)

    N    = length(grid_scat)
    mid  = (N + 1) ÷ 2
    Lmid = electron_electron(
        grid_scat, f0s_scat, mid, mid, [ε_scat], T_scat, Weff_squared, lat_scat
    )
    @test isfinite(Lmid)

    Lend = electron_electron(
        grid_scat, f0s_scat, N, 1, [ε_scat], T_scat, Weff_squared, lat_scat
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
