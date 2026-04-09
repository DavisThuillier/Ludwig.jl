@testset "Permutation Group" begin
    g = PermutationGroupElement([5, 4, 3, 2, 1])
    h = PermutationGroupElement([2, 4, 1, 3, 5])
    gh = PermutationGroupElement([4, 2, 5, 3, 1])

    @test g * h == gh
    @test is_identity(g * inverse(g)) == true
    @test is_identity(h * inverse(h)) == true
    @test is_identity(gh * inverse(gh)) == true

end

@testset "Cyclic Groups" begin
    for n in [1, 2, 3, 4, 6]
        Cₙ = get_cyclic_group(n)
        @test order(Cₙ) == n
    end
end

@testset "Dihedral Groups" begin
    for n in [3, 4, 6]
        Dₙ = get_dihedral_group(n)
        @test order(Dₙ) == 2n
    end
end

@testset "Symmetric Groups" begin
    for n in 1:6
        Sₙ = get_symmetric_group(n)
        @test order(Sₙ) == factorial(n)
    end
end

@testset "Representations" begin
    r₃ = PermutationGroupElement([2, 3, 1]) # Rotation generator in D₃
    s₃ = PermutationGroupElement([1, 3, 2]) # Reflection generator in D₃ fixing point 1
    @test get_matrix_representation(r₃) ≈ [-0.5 -sqrt(3)/2.0; sqrt(3)/2.0 -0.5]
    @test get_matrix_representation(s₃) ≈ [1.0 0.0; 0.0 -1.0]
    @test get_matrix_representation(r₃ * s₃) ≈ [-0.5 -sqrt(3)/2.0; sqrt(3)/2.0 -0.5] * [1.0 0.0; 0.0 -1.0]
    @test get_matrix_representation(s₃ * r₃) ≈ [1.0 0.0; 0.0 -1.0] * [-0.5 -sqrt(3)/2.0; sqrt(3)/2.0 -0.5]

    r₄ = PermutationGroupElement([2, 3, 4, 1]) # Rotation generator in D₄
    s₄ = PermutationGroupElement([1, 4, 3, 2]) # Reflection generator in D₄ fixing point 1
    @test get_matrix_representation(r₄) ≈ [0.0 -1.0; 1.0 0.0]
    @test get_matrix_representation(s₄) ≈ [1.0 0.0; 0.0 -1.0]
    @test get_matrix_representation(r₄ * s₄) ≈ [0.0 -1.0; 1.0 0.0] * [1.0 0.0; 0.0 -1.0]
    @test get_matrix_representation(s₄ * r₄) ≈ [1.0 0.0; 0.0 -1.0] * [0.0 -1.0; 1.0 0.0]

    r₆ = PermutationGroupElement([2, 3, 4, 5, 6, 1]) # Rotation generator in D₆
    s₆ = PermutationGroupElement([1, 6, 5, 4, 3, 2]) # Reflection generator in D₆ fixing point 1
    @test get_matrix_representation(r₆) ≈ [0.5 -sqrt(3)/2.0; sqrt(3)/2.0 0.5]
    @test get_matrix_representation(s₆) ≈ [1.0 0.0; 0.0 -1.0]
    @test get_matrix_representation(r₆ * s₆) ≈ [0.5 -sqrt(3)/2.0; sqrt(3)/2.0 0.5] * [1.0 0.0; 0.0 -1.0]
    @test get_matrix_representation(s₆ * r₆) ≈ [1.0 0.0; 0.0 -1.0] * [0.5 -sqrt(3)/2.0; sqrt(3)/2.0 0.5]
end
