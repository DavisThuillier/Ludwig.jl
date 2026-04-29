using LinearAlgebra

@testset "IBZ symmetry fill: square lattice tight-binding" begin
    # Nearest-neighbor tight-binding on the square lattice at half-filling.
    # The D₄ point group (order 8) means the IBZ is 1/8 of the full BZ.
    T = 0.025
    l = Lattice([1.0 0.0; 0.0 1.0])

    ε(k) = -2.0 * (cos(k[1]) + cos(k[2])) - 1.0 # Nearest neighbor TBM
    bands = [ε]

    Δε    = 0.075  # → n_levels = 2*ceil(α*T/Δε), i.e. n_levels-1 energy patch rows
    n_arc = 4      # n_arc arc-length segments (patches) per row

    mesh = bz_mesh(l, bands, T, Δε, n_arc, 201)
    grid = mesh.patches
    N    = length(grid)

    f0s = f0.(energy.(grid), T)

    Weff_squared(p1, p2, p3, p4; kwargs...) = 1.0 # Hubbard interaction

    sym = bz_symmetry_map(grid, l)

    # --- Structural checks -------------------------------------------------------

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
    ε(k) = -2.0 * (cos(k[1]) + cos(k[2])) - 1.0
    bands = [ε]

    Δε    = 0.075
    n_arc = 4

    mesh = bz_mesh(l, bands, T, Δε, n_arc, 201)
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
