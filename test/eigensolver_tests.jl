using LinearAlgebra
using Random
using StaticArrays

###
### Test matrices
###

function random_hermitian(rng, T, ::Val{N}) where {N}
    A = SMatrix{N,N}(randn(rng, T, N, N))
    return (A + A') / 2
end

# A random unitary rotation of the spectrum 1, 1, 2, 2, ..., so every eigenvalue is doubly
# degenerate (but the last for odd N)
function degenerate_hermitian(rng, T, ::Val{N}) where {N}
    Q = SMatrix{N,N}(Matrix(qr(randn(rng, T, N, N)).Q))
    λ = SVector(ntuple(i -> Float64(cld(i, 2)), Val(N)))
    H = Q * Diagonal(λ) * Q'
    return (H + H') / 2
end

allocated_eigen(H) = @allocated Ludwig.hermitian_eigen(H)
allocated_eigvals(H) = @allocated Ludwig.hermitian_eigvals(H)

rng_eig = MersenneTwister(0x6a61636f)

###
### Tests
###

@testset "hermitian_eigen: agreement with LAPACK" begin
    makers = (random_hermitian, degenerate_hermitian)
    for N in 1:8, T in (Float64, ComplexF64), make in makers
        for _ in 1:10
            H = make(rng_eig, T, Val(N))
            F = Ludwig.hermitian_eigen(H)
            @test F.values isa SVector{N,Float64}
            @test F.vectors isa SMatrix{N,N,T}
            @test isapprox(F.values, eigvals(Hermitian(Matrix(H))); atol=1e-13 * norm(H))
            @test norm(H * F.vectors - F.vectors * Diagonal(F.values)) ≤ 1e-13 * norm(H)
            @test norm(F.vectors' * F.vectors - I) ≤ 1e-13
            @test Ludwig.hermitian_eigvals(H) == F.values
        end
    end
end

@testset "hermitian_eigen: no allocation" begin
    for H in (random_hermitian(rng_eig, ComplexF64, Val(4)),
              random_hermitian(rng_eig, Float64, Val(6)))
        allocated_eigen(H)
        allocated_eigvals(H)
        @test allocated_eigen(H) == 0
        @test allocated_eigvals(H) == 0
    end
end

@testset "hermitian_eigen: edge cases" begin
    # Integer input is promoted to floating point; Float32 stays Float32
    @test Ludwig.hermitian_eigvals(@SMatrix [2 1; 1 2]) ≈ [1.0, 3.0]
    @test Ludwig.hermitian_eigen(@SMatrix [2 1; 1 2]).values ≈ [1.0, 3.0]
    @test Ludwig.hermitian_eigvals(@SMatrix Float32[2 1; 1 2]) isa SVector{2,Float32}

    Z = zero(SMatrix{3,3,ComplexF64})
    @test Ludwig.hermitian_eigen(Z).values == zeros(3)
    @test Ludwig.hermitian_eigen(Z).vectors == I

    # One sweep cannot diagonalize a dense 8×8 matrix
    H = random_hermitian(rng_eig, ComplexF64, Val(8))
    @test_throws ErrorException Ludwig.jacobi_diagonal!(MMatrix(H), nothing, 1)

    # Matrices that are not static go to LAPACK
    Hm = Matrix(H)
    @test Ludwig.hermitian_eigvals(Hm) ≈ eigvals(Hermitian(Hm))
    @test Ludwig.hermitian_eigen(Hm).values ≈ eigvals(Hermitian(Hm))
end
