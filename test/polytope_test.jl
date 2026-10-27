###
### Convex polytopes (src/polytope.jl)
###
#
# The half-space representation on its own: face construction, membership, and clipping a simplex
# against the half-spaces. Its use as a `level_set` mask — culling and snapping marched output —
# is covered in marching_simplices_test.jl.

using Test
using Ludwig
using StaticArrays
using LinearAlgebra

using Ludwig: ConvexPolytope, HalfSpace, in_convex_polytope, clip_segment_to_polytope, clip_triangle_to_polytope!

@testset "polytope helpers (2D)" begin
    # unit square as the intersection of its 4 axis-aligned half-spaces
    square = ConvexPolytope([HalfSpace(SVector( 1.0, 0.0), 1.0), HalfSpace(SVector(-1.0, 0.0), 0.0),
                             HalfSpace(SVector( 0.0, 1.0), 1.0), HalfSpace(SVector( 0.0,-1.0), 0.0)])
    @test length(square.faces) == 4
    for face in square.faces                       # outward unit normals
        @test isapprox(norm(face.normal), 1.0; atol = 1e-12)
    end

    @test in_convex_polytope(SVector(0.5, 0.5), square)    # interior
    @test in_convex_polytope(SVector(0.0, 0.5), square)    # on boundary edge
    @test in_convex_polytope(SVector(1.0, 1.0), square)    # on vertex
    @test !in_convex_polytope(SVector(1.5, 0.5), square)   # outside
    @test !in_convex_polytope(SVector(-1e-9, 0.5), square) # just outside, atol = 0

    # clip a segment against the square
    r1, r2, keep = clip_segment_to_polytope(SVector(0.2, 0.5), SVector(0.8, 0.5), square)
    @test keep && r1 ≈ SVector(0.2, 0.5) && r2 ≈ SVector(0.8, 0.5)   # fully inside: unchanged

    _, _, keep_out = clip_segment_to_polytope(SVector(2.0, 0.5), SVector(3.0, 0.5), square)
    @test !keep_out                                                  # fully outside: dropped

    s1, s2, keep_cross = clip_segment_to_polytope(SVector(0.5, 0.5), SVector(2.0, 0.5), square)
    @test keep_cross && s1 ≈ SVector(0.5, 0.5) && s2 ≈ SVector(1.0, 0.5)  # snapped to edge x = 1
end

@testset "polytope helpers (3D)" begin
    # unit cube as the intersection of its 6 axis-aligned half-spaces
    cube = ConvexPolytope([HalfSpace(SVector( 1.0,0.0,0.0), 1.0), HalfSpace(SVector(-1.0,0.0,0.0), 0.0),
                           HalfSpace(SVector(0.0, 1.0,0.0), 1.0), HalfSpace(SVector(0.0,-1.0,0.0), 0.0),
                           HalfSpace(SVector(0.0,0.0, 1.0), 1.0), HalfSpace(SVector(0.0,0.0,-1.0), 0.0)])
    @test length(cube.faces) == 6
    for face in cube.faces
        @test isapprox(norm(face.normal), 1.0; atol = 1e-12)
    end
    @test in_convex_polytope(SVector(0.5,0.5,0.5), cube)     # interior
    @test in_convex_polytope(SVector(0.0,0.5,0.5), cube)     # on a face
    @test in_convex_polytope(SVector(1.0,1.0,1.0), cube)     # on a vertex
    @test !in_convex_polytope(SVector(1.5,0.5,0.5), cube)    # outside
    @test !in_convex_polytope(SVector(-1e-9,0.5,0.5), cube)  # just outside, atol = 0

    # corner tetrahedron (vertices origin + unit axes) → 4 faces
    tet = ConvexPolytope([HalfSpace(SVector(-1.0,0.0,0.0), 0.0), HalfSpace(SVector(0.0,-1.0,0.0), 0.0),
                          HalfSpace(SVector(0.0,0.0,-1.0), 0.0),
                          HalfSpace(SVector(1.0,1.0,1.0) / sqrt(3.0), 1 / sqrt(3.0))])  # x + y + z ≤ 1
    @test length(tet.faces) == 4

    # clip a surface triangle against the cube
    buffer_A = SVector{3,Float64}[]; buffer_B = SVector{3,Float64}[]
    positive = SVector(0.5,0.5,5.0)   # off the triangle plane, toward +z

    verts = SVector{3,Float64}[]; simps = NTuple{3,Int}[]
    clip_triangle_to_polytope!(verts, simps, SVector(0.2,0.2,0.5), SVector(0.6,0.2,0.5),
                               SVector(0.2,0.6,0.5), positive, cube, buffer_A, buffer_B)
    @test length(simps) == 1                        # fully inside: one triangle
    @test all(in_convex_polytope(v, cube; atol = 1e-9) for v in verts)

    empty!(verts); empty!(simps)
    clip_triangle_to_polytope!(verts, simps, SVector(2.0,2.0,0.5), SVector(3.0,2.0,0.5),
                               SVector(2.0,3.0,0.5), positive, cube, buffer_A, buffer_B)
    @test isempty(simps)                            # fully outside: nothing

    empty!(verts); empty!(simps)
    clip_triangle_to_polytope!(verts, simps, SVector(0.5,0.5,0.5), SVector(2.0,0.5,0.5),
                               SVector(0.5,2.0,0.5), positive, cube, buffer_A, buffer_B)
    @test !isempty(simps)                           # straddling: clipped
    @test all(in_convex_polytope(v, cube; atol = 1e-9) for v in verts)
end
