using LinearAlgebra

@testset "find_contour: unit circle" begin
    xs = LinRange(-1.5, 1.5, 101)
    ys = LinRange(-1.5, 1.5, 101)
    A  = [x^2 + y^2 - 1.0 for x in xs, y in ys]

    bundle = Ludwig.find_contour(xs, ys, A, 0.0)
    @test length(bundle.isolines) == 1
    @test bundle.isolines[1].isclosed
    @test all(p -> isapprox(norm(p), 1.0; atol = 0.05), bundle.isolines[1].points)
    @test isapprox(bundle.isolines[1].arclength, 2π; atol = 0.1)
end

@testset "contour_intersection: unit circle along x-axis" begin
    xs = LinRange(-1.5, 1.5, 101)
    ys = LinRange(-1.5, 1.5, 101)
    A  = [x^2 + y^2 - 1.0 for x in xs, y in ys]

    bundle = Ludwig.find_contour(xs, ys, A, 0.0)
    iso    = bundle.isolines[1]

    p, _ = contour_intersection([0.0, 0.0], [1.0, 0.0], iso)
    @test all(isfinite, p)
    @test isapprox(norm(p), 1.0; atol = 0.05)
end
