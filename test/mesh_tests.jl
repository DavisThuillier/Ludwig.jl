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

###
### Corner-region meshing
###

# Crossing-number test for a point inside a (possibly degenerate) quadrilateral
function in_quad(p, q)
    inside = false
    for i in eachindex(q)
        a, b = q[i], q[mod1(i + 1, length(q))]
        if (a[2] > p[2]) != (b[2] > p[2])
            x = a[1] + (p[2] - a[2]) * (b[1] - a[1]) / (b[2] - a[2])
            x > p[1] && (inside = !inside)
        end
    end
    return inside
end

# Check that the patches of `mesh` tile the triangle `region` and that `ε` stays within
# each patch's energy window: returns (Σ dV / area, uncovered fraction, overlapped
# fraction, fraction of samples outside their patch's energy window).
function corner_mesh_quality(mesh, region, ε; Ng = 200)
    quads = [corners(mesh)[ci] for ci in corner_indices(mesh)]
    area_tri = abs(Ludwig.signed_area(region...))
    (x0, x1), (y0, y1) = get_bounding_box(region)
    n_in = n_gap = n_over = n_out = 0
    for x in LinRange(x0, x1, Ng), y in LinRange(y0, y1, Ng)
        k = [x, y]
        in_polygon(k, region) || continue
        n_in += 1
        hits = findall(q -> in_quad(k, q), quads)
        isempty(hits)     && (n_gap += 1)
        length(hits) ≥ 2  && (n_over += 1)
        for h in hits
            p = patches(mesh)[h]
            abs(ε(k) - p.e) ≤ 0.51 * p.de || (n_out += 1)
        end
    end
    return sum(area, patches(mesh)) / area_tri, n_gap / n_in, n_over / n_in, n_out / n_in
end

@testset "mesh_corner_region: saddle on an edge" begin
    # The edge from K to p2 carries a saddle at S. Its contour splits the triangle into a
    # fan from p1 below the saddle energy and fans to K and to p2 above it, the
    # arrangement found near K in the gapped bilayer-graphene conduction band.
    p1, K, p2 = SVector(0.0, 0.0), SVector(1.0, 0.0), SVector(0.3, 0.9)
    region = [p1, K, p2]
    u = normalize(p2 - K)
    n = SVector(-u[2], u[1])
    n = dot(n, p1 - K) > 0 ? n : -n
    S = K + 0.45 * (p2 - K)
    ε_saddle(k) = dot(k - S, u)^2 - dot(k - S, n)^2 + 0.3 * dot(k - S, n)^3

    mesh = Ludwig.mesh_corner_region(region, ε_saddle, 1, 0.1, 0.1, 401)

    @test length(patches(mesh)) > 0
    @test all(p -> all(isfinite, p.jinv) && isfinite(p.djinv), patches(mesh))
    @test all(p -> p.dV > 0, patches(mesh))
    @test all(p -> in_polygon(p.k, region), patches(mesh))
    @test any(c -> norm(c - S) < 1e-3, corners(mesh)) # Patches meet at the saddle

    area_ratio, gap, over, out = corner_mesh_quality(mesh, region, ε_saddle)
    @test isapprox(area_ratio, 1.0; atol = 0.01)
    @test gap  < 0.005
    @test over < 0.005
    @test out  < 0.005
end

@testset "mesh_corner_region: corner slide without a saddle" begin
    # Monotone on every edge: the contour through the middle vertex splits the triangle
    # into a fan from the lowest vertex and a fan to the highest.
    region = [SVector(0.0, 0.0), SVector(1.0, 0.0), SVector(0.2, 0.8)]
    ε_tilt(k) = k[1] + 2.0 * k[2] + 0.2 * k[1] * k[2]

    mesh = Ludwig.mesh_corner_region(region, ε_tilt, 1, 0.2, 0.2, 401)

    @test length(patches(mesh)) > 0
    @test all(p -> p.dV > 0, patches(mesh))

    area_ratio, gap, over, out = corner_mesh_quality(mesh, region, ε_tilt)
    @test isapprox(area_ratio, 1.0; atol = 0.01)
    @test gap  < 0.005
    @test over < 0.005
    @test out  < 0.005
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
