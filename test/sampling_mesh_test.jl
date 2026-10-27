###
### Decoupled sampling mesh (src/sampling_mesh.jl)
###
#
# The coarse candidate mesh and the fine → coarse map that carries a fine integration measure onto
# it: nearest-simplex location, barycentric interpolation, mass-conserving scatter, and the
# fixed-depth `sampling_mesh` builder.

using Test
using Ludwig
using StaticArrays
using LinearAlgebra

using Ludwig: sampling_mesh, surface_locator, locate, interpolate, scatter_to_vertices
using Ludwig: level_set, centroids, volume, integrate_over_mesh, SimplexMesh

@testset "Decoupled sampling mesh" begin
    sphere(p) = (p[1] - 0.05)^2 + (p[2] - 0.05)^2 + (p[3] - 0.05)^2
    box = ((-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0))

    @testset "locate nearest coarse simplex" begin
        coarse = level_set(sphere, 1.3, box..., 3; refinement_depth = 5, weld = true)
        loc = surface_locator(coarse)
        cents = centroids(coarse)
        # a coarse centroid maps back to its own triangle, at zero distance, with equal weights
        for i in (1, length(cents) ÷ 2, length(cents))
            j, w, d = locate(loc, cents[i])
            @test j == i
            @test d < 1e-9
            @test all(≈(1 / 3), w)
        end
        # every located point lies within a simplex extent of the surface
        @test all(locate(loc, c)[3] < loc.cell for c in cents)
    end

    @testset "barycentric interpolation" begin
        coarse = level_set(sphere, 1.3, box..., 3; refinement_depth = 5, weld = true)
        loc = surface_locator(coarse)
        # a linear field is reproduced exactly at a coarse vertex (all weight on that vertex)
        g(x) = 2.0 * x[1] - 3.0 * x[2] + 0.5 * x[3] + 1.0
        gvals = reshape([g(v) for v in coarse.vertices], :, 1)
        for v in (coarse.vertices[1], coarse.vertices[end])
            @test interpolate(loc, gvals, v)[1] ≈ g(v)
        end
        # a fine-surface point interpolates the linear field to its value at the closest coarse point:
        # the error is exactly ∇g·(q - p), bounded by |∇g| times the located distance.
        grad_g = SVector(2.0, -3.0, 0.5)
        fine = level_set(sphere, 1.3, box..., 4; refinement_depth = 7, weld = true)
        for p in centroids(fine)[1:200:end]
            _, _, d = locate(loc, p)
            @test abs(interpolate(loc, gvals, p)[1] - g(p)) ≤ norm(grad_g) * d + 1e-9
        end
    end

    @testset "locate and interpolate on a contour (2D)" begin
        # The D = 2 path, where the closest point lies on a segment rather than in a triangle.
        # Off-grid center and level, as in 3D, so the contour avoids lattice-node degeneracies.
        circle(p) = (p[1] - 0.05)^2 + (p[2] - 0.05)^2
        coarse = level_set(circle, 1.3, (-2.0, 2.0), (-2.0, 2.0), 3; refinement_depth = 5, weld = true)
        loc = surface_locator(coarse)
        cents = centroids(coarse)
        # a coarse centroid maps back to its own segment, at zero distance, with equal weights
        for i in (1, length(cents) ÷ 2, length(cents))
            j, w, d = locate(loc, cents[i])
            @test j == i
            @test d < 1e-9
            @test all(≈(1 / 2), w)
        end
        @test all(locate(loc, c)[3] < loc.cell for c in cents)

        g(x) = 1.5 * x[1] - 0.75 * x[2] + 0.25
        gvals = reshape([g(v) for v in coarse.vertices], :, 1)
        for v in (coarse.vertices[1], coarse.vertices[end])
            @test interpolate(loc, gvals, v)[1] ≈ g(v)
        end
        grad_g = SVector(1.5, -0.75)
        fine = level_set(circle, 1.3, (-2.0, 2.0), (-2.0, 2.0), 4; refinement_depth = 7, weld = true)
        for p in centroids(fine)[1:5:end]
            _, _, d = locate(loc, p)
            @test abs(interpolate(loc, gvals, p)[1] - g(p)) ≤ norm(grad_g) * d + 1e-9
        end
    end

    @testset "mass-conserving scatter" begin
        coarse = level_set(sphere, 1.3, box..., 3; refinement_depth = 5, weld = true)
        fine   = level_set(sphere, 1.3, box..., 4; refinement_depth = 7, weld = true)
        loc = surface_locator(coarse)
        fine_centroids = centroids(fine)
        fine_masses = [volume(i, fine) for i in eachindex(fine.simplices)]   # ρ ≡ 1
        transferred = scatter_to_vertices(loc, fine_centroids, fine_masses)
        # total mass is conserved onto the coarse vertices (weights sum to one per point)
        @test sum(transferred) ≈ sum(fine_masses)
        # and equals the fine surface area
        @test sum(transferred) ≈ integrate_over_mesh(_ -> 1.0, fine)
    end

    @testset "sampling_mesh fixed depth" begin
        # a fixed sampling_depth builds exactly the welded level_set at that depth — no adaptive escalation
        fixed = sampling_mesh(sphere, 1.3, box..., 3; sampling_depth = 5)
        reference = level_set(sphere, 1.3, box..., 3; refinement_depth = 5, weld = true)
        @test length(fixed.simplices) == length(reference.simplices)
        @test length(fixed.vertices) == length(reference.vertices)
        @test fixed isa SimplexMesh{3}

        # deterministic: rebuilding at the same depth reproduces the candidate set exactly
        again = sampling_mesh(sphere, 1.3, box..., 3; sampling_depth = 5)
        @test fixed.vertices == again.vertices
        @test fixed.simplices == again.simplices

        # sampling_depth below initial_depth is rejected, matching level_set's own depth validation
        @test_throws ArgumentError sampling_mesh(sphere, 1.3, box..., 3; sampling_depth = 2)
    end
end
