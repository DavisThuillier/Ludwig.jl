using Ludwig

@testset "Circular Fermi Surface Inversion Symmetry Test" begin
    n_ε = 12
    n_θ = 201
    T = 0.0075
    α = 12.0

    m = 1.0 # Really ħ^-2 m ε_F
    ε(k) = k^2 / (2 * m) - 1.0
    
    mesh = FSMesh.circular_fs_mesh(ε, T, n_ε, n_θ, α)
    ℓ = length(mesh.patches)

    for j ∈ 1:(n_θ-1)÷2
        for i ∈ 1:n_ε-1
            p1 = mesh.patches[(j-1) * (n_ε-1) + i]
            p2 = mesh.patches[(n_θ-j-1) * (n_ε-1) + i]
            @test p1.e  == p2.e
            @test p1.de == p2.de
            @test p1.dV == p2.dV
            @test p1.djinv ≈ p2.djinv
            @test abs((p1.v .- p2.v)[1]) < 1e-14
            @test abs((p1.v .+ p2.v)[2]) < 1e-14
        end
    end
end
