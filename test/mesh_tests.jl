using LinearAlgebra

###
### Shared workload
###

const T_mesh     = 0.005
const Δε_mesh    = 1.5 * T_mesh
const n_arc_mesh = 6
const N_grid_mesh = 401
const lat_mesh   = Lattice([1.0 0.0; 0.0 1.0])
ε_mesh(k) = -2.0 * (cos(k[1]) + cos(k[2]))

@testset "ibz_mesh: square lattice tight-binding" begin
    mesh = ibz_mesh(lat_mesh, [ε_mesh], T_mesh, Δε_mesh, n_arc_mesh, N_grid_mesh)
    ibz  = get_ibz(lat_mesh)

    @test isa(mesh, Mesh)
    @test length(patches(mesh)) > 0
    @test all(p -> isfinite(p.e),            patches(mesh))
    @test all(p -> all(isfinite, p.k),       patches(mesh))
    @test all(p -> all(isfinite, p.v),       patches(mesh))
    @test all(p -> isfinite(p.de),           patches(mesh))
    @test all(p -> isfinite(p.dV),           patches(mesh))
    @test all(p -> all(isfinite, p.jinv),    patches(mesh))
    @test all(p -> isfinite(p.djinv),        patches(mesh))
    @test all(p -> p.dV > 0,                 patches(mesh))
    @test all(p -> in_polygon(p.k, ibz),     patches(mesh))
end

@testset "bz_mesh: square lattice tight-binding" begin
    mesh_ibz = ibz_mesh(lat_mesh, [ε_mesh], T_mesh, Δε_mesh, n_arc_mesh, N_grid_mesh)
    mesh_bz  = bz_mesh(lat_mesh,  [ε_mesh], T_mesh, Δε_mesh, n_arc_mesh, N_grid_mesh)
    G_order  = order(point_group(lat_mesh))

    @test isa(mesh_bz, Mesh)
    @test length(patches(mesh_bz)) == G_order * length(patches(mesh_ibz))
    @test all(p -> isfinite(p.e),         patches(mesh_bz))
    @test all(p -> all(isfinite, p.k),    patches(mesh_bz))
    @test all(p -> all(isfinite, p.v),    patches(mesh_bz))
    @test all(p -> isfinite(p.de),        patches(mesh_bz))
    @test all(p -> isfinite(p.dV),        patches(mesh_bz))
    @test all(p -> all(isfinite, p.jinv), patches(mesh_bz))
    @test all(p -> isfinite(p.djinv),     patches(mesh_bz))
    @test all(p -> p.dV > 0,              patches(mesh_bz))

    # Regression: bz_symmetry_map must accept the BZ mesh produced by `bz_mesh`.
    bzmap = bz_symmetry_map(patches(mesh_bz), lat_mesh)
    @test isa(bzmap, BZSymmetryMap)
    @test length(bzmap.ibz_inds) == length(patches(mesh_ibz))
end

@testset "mesh_region: IBZ polygon of square lattice" begin
    ibz   = get_ibz(lat_mesh)
    e_min = -6.0 * T_mesh
    e_max =  6.0 * T_mesh
    mesh  = Ludwig.mesh_region(ibz, ε_mesh, e_min, e_max, Δε_mesh, n_arc_mesh, N_grid_mesh)

    @test isa(mesh, Mesh)
    @test length(patches(mesh)) > 0
    @test all(p -> isfinite(p.e),         patches(mesh))
    @test all(p -> all(isfinite, p.k),    patches(mesh))
    @test all(p -> all(isfinite, p.v),    patches(mesh))
    @test all(p -> isfinite(p.de),        patches(mesh))
    @test all(p -> isfinite(p.dV),        patches(mesh))
    @test all(p -> all(isfinite, p.jinv), patches(mesh))
    @test all(p -> isfinite(p.djinv),     patches(mesh))
    @test all(p -> p.dV > 0,              patches(mesh))
end

@testset "isotropic_mesh: parabolic radial dispersion" begin
    ε_iso(r) = 0.5 * (r^2 - 1.0)
    kf = [1.0]
    mesh = isotropic_mesh(ε_iso, kf, T_mesh, Δε_mesh, n_arc_mesh)

    @test isa(mesh, Mesh)
    @test length(patches(mesh)) > 0
    @test all(p -> isfinite(p.e),         patches(mesh))
    @test all(p -> all(isfinite, p.k),    patches(mesh))
    @test all(p -> all(isfinite, p.v),    patches(mesh))
    @test all(p -> isfinite(p.de),        patches(mesh))
    @test all(p -> isfinite(p.dV),        patches(mesh))
    @test all(p -> all(isfinite, p.jinv), patches(mesh))
    @test all(p -> isfinite(p.djinv),     patches(mesh))
    @test all(p -> p.dV > 0,              patches(mesh))
    @test all(p -> isapprox(norm(p.k), kf[1]; rtol = 0.1), patches(mesh))
end
