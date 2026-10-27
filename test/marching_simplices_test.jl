###
### Marching simplices (src/marching_simplices.jl)
###
#
# `level_set` in 2D (marching triangles) and 3D (marching tetrahedra): contour/surface accuracy,
# exactness on a linear field, empty levels, depth validation, the caller-supplied sample buffer,
# vertex welding, and culling/snapping against a `ConvexPolytope` mask. The mask type itself is
# tested in polytope_test.jl.

using Test
using Ludwig
using StaticArrays
using LinearAlgebra

using Ludwig: level_set, merge_vertices, SimplexMesh, integrate_over_mesh
using Ludwig: ConvexPolytope, HalfSpace, in_convex_polytope

# Total (D-1)-measure of a mesh: arclength for a contour, area for a surface.
contour_length(mesh) = integrate_over_mesh(_ -> 1.0, mesh)
surface_area(mesh) = integrate_over_mesh(_ -> 1.0, mesh)

@testset "Marching Triangles" begin

    @testset "invalid initial_depth" begin
        f(p) = p[1]
        @test_throws ArgumentError level_set(f, 0.0, (-2.0, 2.0), (-2.0, 2.0), -1)
        @test_throws ArgumentError level_set(f, 0.0, (-2.0, 2.0), (-2.0, 2.0), 3; refinement_depth = 2)
    end

    @testset "circle contour" begin
        # Off-grid center and level so the contour avoids lattice-node degeneracies;
        # N must be fine enough that the curvature within a seed triangle is resolved
        # (corner-only sign detection misses sub-seed-triangle arcs by design).
        f(p) = (p[1] - 0.05)^2 + (p[2] - 0.05)^2
        lvl = 1.3
        r = sqrt(lvl)
        mesh = level_set(f, lvl, (-2.0, 2.0), (-2.0, 2.0), 5; refinement_depth = 7)
        @test mesh isa SimplexMesh{2}
        @test !isempty(mesh.simplices)

        # every emitted endpoint lies on the circle to O(h^2)
        for v in mesh.vertices
            @test abs(f(v) - lvl) < 0.005
        end

        # total length approximates the circumference 2πr
        @test isapprox(contour_length(mesh), 2π * r; atol = 1e-2)
    end

    @testset "linear plane" begin
        f(p) = p[1]
        mesh = level_set(f, 0.0, (-2.0, 2.0), (-2.0, 2.0), 2; refinement_depth = 7)
        @test !isempty(mesh.simplices)
        # linear interpolation is exact: the contour is the line x = 0
        for v in mesh.vertices
            @test abs(v[1]) < 1e-12
        end
    end

    @testset "empty contour" begin
        f(p) = p[1]^2 + p[2]^2
        mesh = level_set(f, -1.0, (-2.0, 2.0), (-2.0, 2.0), 2; refinement_depth = 7)   # level below the range of f
        @test isempty(mesh.simplices)
        @test isempty(mesh.vertices)
    end

    @testset "weld" begin
        f(p) = (p[1] - 0.05)^2 + (p[2] - 0.05)^2
        soup   = level_set(f, 1.3, (-2.0, 2.0), (-2.0, 2.0), 4; refinement_depth = 7)
        welded = level_set(f, 1.3, (-2.0, 2.0), (-2.0, 2.0), 4; refinement_depth = 7, weld = true)

        # the "segment soup" carries two fresh vertices per segment
        @test length(soup.vertices) == 2 * length(soup.simplices)
        # welding does not change the segments themselves
        @test length(welded.simplices) == length(soup.simplices)
        @test isapprox(contour_length(welded), contour_length(soup); atol = 1e-12)
        # a single closed loop: every interior vertex shared by exactly two segments
        @test length(welded.vertices) == length(welded.simplices)
        @test length(welded.vertices) < length(soup.vertices)
    end

    @testset "caller-supplied sample buffer" begin
        f(p) = p[1]^2 + p[2]^2
        initial_depth = 2
        N = 2^initial_depth + 1
        buffer = Matrix{Float64}(undef, N, N)
        mesh = level_set(f, 1.0, (-2.0, 2.0), (-2.0, 2.0), initial_depth; refinement_depth = 2, samples = buffer)
        @test !isempty(mesh.simplices)
        # the buffer holds the shifted grid samples g = f - level
        @test buffer[1, 1] ≈ f(SVector(-2.0, -2.0)) - 1.0
        @test buffer[3, 3] ≈ f(SVector(0.0, 0.0)) - 1.0
    end

    @testset "merge_vertices utility" begin
        # two coincident-within-atol vertices and one distinct vertex
        verts = [SVector(0.0, 0.0), SVector(1e-9, 0.0), SVector(1.0, 0.0)]
        simps = [(1, 3), (2, 3)]
        mesh = SimplexMesh{2}(verts, simps)
        merged = merge_vertices(mesh; atol = 1e-6)
        @test length(merged.vertices) == 2
        @test merged.simplices[1][1] == merged.simplices[2][1]  # both welded to the same vertex
    end

    @testset "polytope culling + snapping (2D)" begin
        f(p) = (p[1] - 0.05)^2 + (p[2] - 0.05)^2
        lvl = 1.3
        # a triangular wedge (vertices (2,-2), (-2,2), (-2,-2)) that clips part of the circle
        poly = ConvexPolytope([HalfSpace(SVector(1.0, 1.0) / sqrt(2.0), 0.0),   # x + y ≤ 0
                               HalfSpace(SVector(-1.0, 0.0), 2.0),              # x ≥ -2
                               HalfSpace(SVector(0.0, -1.0), 2.0)])             # y ≥ -2

        full = level_set(f, lvl, (-2.0, 2.0), (-2.0, 2.0), 5; refinement_depth = 7)
        cut  = level_set(f, lvl, (-2.0, 2.0), (-2.0, 2.0), 5; refinement_depth = 7, mask = poly)

        @test cut isa SimplexMesh{2}
        @test !isempty(cut.simplices)
        @test length(cut.simplices) < length(full.simplices)   # culling dropped the outside half

        for v in cut.vertices
            @test in_convex_polytope(v, poly; atol = 1e-9)      # snapping keeps every vertex inside
            # every vertex lies on the circle (a contour crossing) or on the polytope boundary (a snap)
            on_circle   = abs(f(v) - lvl) < 0.005
            on_boundary = !in_convex_polytope(v, poly; atol = -1e-7)  # not strictly interior ⇒ on an edge
            @test on_circle || on_boundary
        end

        # a polytope enclosing the whole box reproduces the unculled mesh (fully-interior fast path)
        big = ConvexPolytope([HalfSpace(SVector( 1.0, 0.0), 10.0), HalfSpace(SVector(-1.0, 0.0), 10.0),
                              HalfSpace(SVector( 0.0, 1.0), 10.0), HalfSpace(SVector( 0.0,-1.0), 10.0)])
        enclosed = level_set(f, lvl, (-2.0, 2.0), (-2.0, 2.0), 5; refinement_depth = 7, mask = big)
        @test length(enclosed.simplices) == length(full.simplices)

        # explicit nothing reproduces the baseline
        @test length(level_set(f, lvl, (-2.0, 2.0), (-2.0, 2.0), 5; refinement_depth = 7, mask = nothing).simplices) ==
              length(full.simplices)
    end

end

@testset "Marching Tetrahedra" begin

    @testset "sphere surface" begin
        # off-grid center and level to avoid lattice-node degeneracies
        f(p) = (p[1] - 0.05)^2 + (p[2] - 0.05)^2 + (p[3] - 0.05)^2
        lvl = 1.3
        r = sqrt(lvl)
        mesh = level_set(f, lvl, (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), 4; refinement_depth = 6)
        @test mesh isa SimplexMesh{3}
        @test !isempty(mesh.simplices)

        for v in mesh.vertices
            @test abs(f(v) - lvl) < 0.01
        end
        @test isapprox(surface_area(mesh), 4π * r^2; rtol = 1e-2)
    end

    @testset "plane" begin
        f(p) = p[1]
        mesh = level_set(f, 0.0, (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), 3; refinement_depth = 4)
        @test !isempty(mesh.simplices)
        for v in mesh.vertices
            @test abs(v[1]) < 1e-12   # linear: surface is exactly the plane x = 0
        end
    end

    @testset "empty surface" begin
        f(p) = p[1]^2 + p[2]^2 + p[3]^2
        mesh = level_set(f, -1.0, (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), 3; refinement_depth = 4)
        @test isempty(mesh.simplices)
        @test isempty(mesh.vertices)
    end

    @testset "weld closed surface" begin
        f(p) = (p[1] - 0.05)^2 + (p[2] - 0.05)^2 + (p[3] - 0.05)^2
        soup   = level_set(f, 1.3, (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), 4; refinement_depth = 7)
        welded = level_set(f, 1.3, (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), 4; refinement_depth = 7, weld = true)

        F = length(soup.simplices)
        @test length(soup.vertices) == 3F                 # three fresh vertices per triangle
        @test length(welded.simplices) == F
        @test length(welded.vertices) < length(soup.vertices)
        # closed triangulated surface: V - E + F = 2 with E = 3F/2 ⇒ V = F/2 + 2,
        # up to a handful of degenerate triangles where the surface grazes lattice nodes
        @test abs(length(welded.vertices) - (F ÷ 2 + 2)) < 1e-3 * F
        @test isapprox(surface_area(welded), surface_area(soup); atol = 1e-9)
    end

    @testset "caller-supplied sample buffer" begin
        f(p) = p[1]^2 + p[2]^2 + p[3]^2
        initial_depth = 2
        N = 2^initial_depth + 1
        buffer = Array{Float64,3}(undef, N, N, N)
        mesh = level_set(f, 1.0, (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), initial_depth; refinement_depth = 2, samples = buffer)
        @test !isempty(mesh.simplices)
        @test buffer[1, 1, 1] ≈ f(SVector(-2.0, -2.0, -2.0)) - 1.0
        @test buffer[3, 3, 3] ≈ f(SVector(0.0, 0.0, 0.0)) - 1.0
    end

    @testset "invalid initial_depth" begin
        f(p) = p[1]
        @test_throws ArgumentError level_set(f, 0.0, (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), -1)
        @test_throws ArgumentError level_set(f, 0.0, (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), 3; refinement_depth = 2)
    end

    @testset "polytope culling + snapping (3D)" begin
        f(p) = (p[1] - 0.05)^2 + (p[2] - 0.05)^2 + (p[3] - 0.05)^2
        lvl = 1.3
        # a tetrahedron whose slanted face (x+y+z = 0) cuts through the sphere
        # (vertices (4,-2,-2), (-2,4,-2), (-2,-2,4), (-2,-2,-2))
        poly = ConvexPolytope([HalfSpace(SVector(1.0,1.0,1.0) / sqrt(3.0), 0.0),  # x + y + z ≤ 0
                               HalfSpace(SVector(-1.0,0.0,0.0), 2.0),             # x ≥ -2
                               HalfSpace(SVector(0.0,-1.0,0.0), 2.0),             # y ≥ -2
                               HalfSpace(SVector(0.0,0.0,-1.0), 2.0)])            # z ≥ -2

        full = level_set(f, lvl, (-2.0,2.0), (-2.0,2.0), (-2.0,2.0), 4; refinement_depth = 6)
        cut  = level_set(f, lvl, (-2.0,2.0), (-2.0,2.0), (-2.0,2.0), 4; refinement_depth = 6, mask = poly)

        @test cut isa SimplexMesh{3}
        @test !isempty(cut.simplices)
        @test length(cut.simplices) < length(full.simplices)   # culling dropped the outside part

        for v in cut.vertices
            @test in_convex_polytope(v, poly; atol = 1e-9)      # snapping keeps every vertex inside
            on_sphere = abs(f(v) - lvl) < 0.01
            on_face   = !in_convex_polytope(v, poly; atol = -1e-7)  # not strictly interior ⇒ on a face
            @test on_sphere || on_face
        end

        # a polytope enclosing the whole box reproduces the unmasked mesh (fully-interior fast path)
        big = ConvexPolytope([HalfSpace(SVector( 1.0,0.0,0.0), 10.0), HalfSpace(SVector(-1.0,0.0,0.0), 10.0),
                              HalfSpace(SVector(0.0, 1.0,0.0), 10.0), HalfSpace(SVector(0.0,-1.0,0.0), 10.0),
                              HalfSpace(SVector(0.0,0.0, 1.0), 10.0), HalfSpace(SVector(0.0,0.0,-1.0), 10.0)])
        enclosed = level_set(f, lvl, (-2.0,2.0), (-2.0,2.0), (-2.0,2.0), 4; refinement_depth = 6, mask = big)
        @test length(enclosed.simplices) == length(full.simplices)

        # explicit nothing reproduces the baseline
        @test length(level_set(f, lvl, (-2.0,2.0), (-2.0,2.0), (-2.0,2.0), 4;
                               refinement_depth = 6, mask = nothing).simplices) == length(full.simplices)
    end

end
