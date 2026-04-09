using LinearAlgebra

@testset "IBZ symmetry fill: square lattice tight-binding" begin
    # Nearest-neighbor tight-binding on the square lattice at half-filling.
    # The D₄ point group (order 8) means the IBZ is 1/8 of the full BZ.
    T = 0.025
    l = Lattice([1.0 0.0; 0.0 1.0])

    ε(k) = -2.0 * (cos(2π * k[1]) + cos(2π * k[2])) # Nearest neighbor TBM
    bands = [ε]

    n_levels = 4   # (n_levels - 1) = 3 energy patches per sector
    n_cuts   = 4   # (n_cuts   - 1) = 3 angular patches per sector
    n_ibz    = (n_levels - 1) * (n_cuts - 1)   # 9 IBZ patches per band

    mesh = bz_mesh(l, bands, T, n_levels, n_cuts, 201)
    grid = mesh.patches
    N    = length(grid)

    @test N == 8 * n_ibz   # D₄ has order 8

    f0s = f0.(energy.(grid), T)

    Weff_squared(p1, p2, p3, p4; kwargs...) = 1.0 # Hubbard interaction

    sym = bz_symmetry_map(grid, l)

    # --- Structural checks -------------------------------------------------------

    # IBZ should contain exactly n_ibz patches (one per IBZ sector per band)
    @test length(sym.ibz_inds) == n_ibz

    # Every non-IBZ patch must appear exactly once in ibz_preimage
    @test length(sym.ibz_preimage) == N - n_ibz

    # IBZ patches and non-IBZ patches must be disjoint and cover 1:N
    @test isempty(intersect(Set(sym.ibz_inds), Set(keys(sym.ibz_preimage))))
    @test union(Set(sym.ibz_inds), Set(keys(sym.ibz_preimage))) == Set(1:N)

    # IBZ patches must lie inside the IBZ polygon
    ibz = get_ibz(l)
    @test all(i -> in_polygon(grid[i].k, ibz), sym.ibz_inds)

    # --- Correctness check -------------------------------------------------------

    # Full collision matrix (naive: every (i,j) pair)
    L_full = zeros(N, N)
    for i in 1:N, j in 1:N
        L_full[i, j] = electron_electron(grid, f0s, i, j, bands, T, Weff_squared, l)
    end

    # Symmetry-reduced: compute only IBZ rows, then propagate by fill_from_ibz!
    L_sym = zeros(N, N)
    for i in sym.ibz_inds, j in 1:N
        L_sym[i, j] = electron_electron(grid, f0s, i, j, bands, T, Weff_squared, l)
    end
    fill_from_ibz!(L_sym, sym)

    @test maximum(abs, L_full - L_sym) < 1e-10
end

@testset "IBZ diagonalize: eigenvalues match fill_from_ibz! + eigen" begin
    T = 0.025
    l = Lattice([1.0 0.0; 0.0 1.0])
    ε(k) = -2.0 * (cos(2π * k[1]) + cos(2π * k[2]))
    bands = [ε]

    n_levels = 4
    n_cuts   = 4

    mesh = bz_mesh(l, bands, T, n_levels, n_cuts, 201)
    grid = mesh.patches
    N    = length(grid)
    f0s  = f0.(energy.(grid), T)
    Weff_squared(p1, p2, p3, p4; kwargs...) = 1.0

    sym = bz_symmetry_map(grid, l)

    # Populate only IBZ rows
    L_sym = zeros(N, N)
    for i in sym.ibz_inds, j in 1:N
        L_sym[i, j] = electron_electron(grid, f0s, i, j, bands, T, Weff_squared, l)
    end

    # Reference: fill then eigen
    L_ref    = copy(L_sym)
    fill_from_ibz!(L_ref, sym)
    ref_vals = sort(real(eigen(L_ref).values))

    # Diagonalize without fill
    result   = diagonalize_ibz(L_sym, sym)
    new_vals = sort(real(result.values))

    @test length(new_vals) == N
    @test maximum(abs, new_vals - ref_vals) < 1e-10
end
