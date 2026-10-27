###
### SimplexMesh geometry (src/mesh.jl)
###
#
# `perimeter` measures the seam where a clipped surface meets the faces of its bounding polytope.
# An edge counts only when both endpoints lie on a *common* face, so chords spanning two faces at a
# corner, interior refinement cracks, and closed surfaces contribute nothing.

using Test
using Ludwig
using StaticArrays
using LinearAlgebra

using Ludwig: level_set, perimeter, SimplexMesh, ConvexPolytope, HalfSpace

@testset "Mesh perimeter" begin

    @testset "contour boundary is null" begin
        f(p) = p[1]^2 + p[2]^2
        contour = level_set(f, 1.0, (-2.0, 2.0), (-2.0, 2.0), 4; refinement_depth = 6, weld = true)
        square = ConvexPolytope([HalfSpace(SVector( 1.0, 0.0), 2.0), HalfSpace(SVector(-1.0, 0.0), 2.0),
                                 HalfSpace(SVector( 0.0, 1.0), 2.0), HalfSpace(SVector( 0.0,-1.0), 2.0)])
        @test perimeter(contour) === 0.0
        @test perimeter(contour, square) === 0.0    # the polytope argument is ignored in 2D
    end

    @testset "same-face membership and corner guard" begin
        # unit cube [0,1]^3. Edge A-B lies on face x = 0 (length 0.6, counted); edge A-D has one
        # endpoint on x = 0 and the other on y = 0 (no shared face, not counted).
        cube = ConvexPolytope([HalfSpace(SVector( 1.0,0.0,0.0), 1.0), HalfSpace(SVector(-1.0,0.0,0.0), 0.0),
                               HalfSpace(SVector(0.0, 1.0,0.0), 1.0), HalfSpace(SVector(0.0,-1.0,0.0), 0.0),
                               HalfSpace(SVector(0.0,0.0, 1.0), 1.0), HalfSpace(SVector(0.0,0.0,-1.0), 0.0)])
        A = SVector(0.0,0.2,0.5); B = SVector(0.0,0.8,0.5); C = SVector(0.5,0.5,0.5); D = SVector(0.2,0.0,0.5)
        mesh = SimplexMesh{3}([A, B, C, D], [(1,2,3), (1,4,3)])
        @test perimeter(mesh, cube) ≈ 0.6
    end

    @testset "clipped level-set seam length" begin
        # plane z = 0.05 clipped to the cube |x|,|y|,|z| ≤ 1: the seam is the boundary of the 2×2
        # square in that plane, lying on the four side faces (perimeter 8). The z-faces are never
        # reached by the surface, and the cube corners sit on grid nodes so the seam is exact.
        f(p) = p[3]
        cube = ConvexPolytope([HalfSpace(SVector( 1.0,0.0,0.0), 1.0), HalfSpace(SVector(-1.0,0.0,0.0), 1.0),
                               HalfSpace(SVector(0.0, 1.0,0.0), 1.0), HalfSpace(SVector(0.0,-1.0,0.0), 1.0),
                               HalfSpace(SVector(0.0,0.0, 1.0), 1.0), HalfSpace(SVector(0.0,0.0,-1.0), 1.0)])
        cut = level_set(f, 0.05, (-2.0,2.0), (-2.0,2.0), (-2.0,2.0), 3; refinement_depth = 5, weld = true, mask = cube)
        @test perimeter(cut, cube) ≈ 8.0
    end

    @testset "closed non-monotonic surface has no seam" begin
        # A sphere has an interior minimum (non-monotonic dispersion), the case where the earlier
        # edge-multiplicity test inflated the perimeter with interior refinement cracks. Fully inside
        # the box the surface is closed, so the geometric test correctly returns zero: no crack edge
        # lies on a box face.
        f(p) = (p[1]-0.05)^2 + (p[2]-0.05)^2 + (p[3]-0.05)^2
        sphere = level_set(f, 1.3, (-2.0,2.0), (-2.0,2.0), (-2.0,2.0), 4; refinement_depth = 7, weld = true)
        box = ConvexPolytope([HalfSpace(SVector( 1.0,0.0,0.0), 2.0), HalfSpace(SVector(-1.0,0.0,0.0), 2.0),
                              HalfSpace(SVector(0.0, 1.0,0.0), 2.0), HalfSpace(SVector(0.0,-1.0,0.0), 2.0),
                              HalfSpace(SVector(0.0,0.0, 1.0), 2.0), HalfSpace(SVector(0.0,0.0,-1.0), 2.0)])
        @test perimeter(sphere, box) == 0.0
    end

end
