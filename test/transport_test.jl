###
### Transport coefficients (src/transport.jl)
###

using Test
using Ludwig
using StaticArrays
using LinearAlgebra

using Ludwig: gausslegendre
using Ludwig: boltzmann_weight, prepare_solve, detailed_balance_factors
using Ludwig: thermal_conductivity, thermoelectric_conductivity, peltier_tensor

@isdefined(cubic_crystal) || include("crystal_fixtures.jl")

@testset "Transport coefficients" begin

    # Quadrature rule for the 2D parabolic band ε = k²/2m - μ. In polar coordinates
    # d²k/(2π)² = (m/(2π)²) dε dθ, so an angular midpoint rule crossed with Gauss-Legendre in energy
    # is an exact-to-roundoff full-BZ rule for this band, with weights in the package's d^Dk/(2π)^D
    # convention.
    function parabolic_band_rule(m, μ, T; n_angle = 12, n_energy = 80, window = 20.0)
        x, energy_weights = gausslegendre(n_energy)
        half_width = window * T
        dθ = 2π / n_angle
        v = SVector{2,Float64}[]
        energies = Float64[]
        weights = Float64[]
        for a in 1:n_energy, b in 1:n_angle
            ε = half_width * x[a]
            θ = (b - 1) * dθ
            k = sqrt(2m * (ε + μ))
            push!(v, (k / m) * SVector(cos(θ), sin(θ)))
            push!(energies, ε)
            push!(weights, (m / (2π)^2) * (half_width * energy_weights[a]) * dθ)
        end
        return v, energies, weights
    end

    @testset "Fermi-window weight is D/T" begin
        weights  = rand(20) .+ 0.5
        energies = randn(20) ./ 40
        T = 0.01
        @test boltzmann_weight(energies, weights, T) == detailed_balance_factors(weights, energies, T) ./ T
    end

    @testset "Drude prefactor audit" begin
        # For the relaxation-time operator L = I/τ on a 2D parabolic band the conductivity has the
        # closed form σxx = G0·τ·μ (the mass cancels), equivalently the Drude value n e²τ/(m ħ). This
        # pins the 2π·G0 prefactor against the (2π)^D already carried by the quadrature weights: the
        # bare-measure prefactor G0/(2π) would miss by (2π)² ≈ 39.5.
        m, μ, T, τ = 1.7, 0.35, 0.004, 12.0
        v, energies, weights = parabolic_band_rule(m, μ, T)
        N = length(v)
        L = Matrix{Float64}(I / τ, N, N)

        σ = conductivity(L, v, energies, weights, T)
        @test σ isa SMatrix{2,2,Float64}
        @test isapprox(σ[1, 1], G0 * τ * μ; rtol = 1e-6)
        @test isapprox(σ[2, 2], σ[1, 1]; rtol = 1e-12)
        @test abs(σ[1, 2]) < 1e-12 * σ[1, 1]           # isotropic band: no Hall/off-diagonal response

        # AC response of the same operator is the Drude form σ(ω) = σ_DC / (1 - iωτ), which fixes the
        # sign convention of the -iω shift.
        ω = 0.5 / τ
        σ_ac = conductivity(L, v, energies, weights, T; ω)
        @test isapprox(σ_ac[1, 1], σ[1, 1] / (1 - im * ω * τ); rtol = 1e-10)

        # Finite q with a diagonal L stays diagonal, so the +i q·v shift can be summed by hand.
        q = SVector(0.3, 0.0)
        weight = boltzmann_weight(energies, weights, T)
        vx = getindex.(v, 1)
        expected = (2π * G0) * sum(weight[i] * vx[i]^2 / (1 / τ + im * dot(q, v[i])) for i in 1:N)
        @test isapprox(conductivity(L, v, energies, weights, T; q)[1, 1], expected; rtol = 1e-10)

        # A precomputed factorization is reused, not refactorized, and gives the same answer.
        @test prepare_solve(lu(L), \) isa LinearAlgebra.LU
        @test conductivity(lu(L), v, energies, weights, T) ≈ σ
        @test conductivity(L, v, energies, weights, T; solve = (A, b) -> A \ b) ≈ σ
    end

    @testset "point-group covariance of σ" begin
        # The acceptance gate Auguste specified during the transport design round, kept as its own
        # sub-testset rather than folded into the prefactor audit above: it constrains the symmetry of
        # the response tensor, not the measure convention. Full-BZ nodes are a point-group orbit and
        # L_full is group-invariant, so R·σ·Rᵀ = σ holds exactly. This is a diagnostic, never
        # something the code enforces.
        pg = PointGroup([c4_planar()], square_primitives())
        N_ibz = 5
        N = order(pg) * N_ibz
        mult = group_multiplication_table(operations(pg))

        ibz_nodes = [SVector(0.3 + 0.1r, 0.05 + 0.07r) for r in 1:N_ibz]
        nodes = [operations(pg)[g] * ibz_nodes[r] for g in 1:order(pg) for r in 1:N_ibz]
        v = nodes                                            # v(k) = k is exactly covariant
        energies = [sum(abs2, k) - 0.3 for k in nodes]       # invariant, so orbit-constant
        weights = repeat(rand(N_ibz) .+ 0.5, order(pg))

        # Group-invariant, invertible operator: orbit-symmetrize a random matrix, then shift.
        R0 = randn(N, N)
        L_full = zeros(N, N)
        for a in 1:order(pg)
            perm = grid_permutation(mult, a, N_ibz)
            L_full .+= R0[perm, perm]
        end
        L_full += N * I

        σ = conductivity(L_full, v, energies, weights, 0.02)
        for R in operations(pg)
            @test norm(R * σ * transpose(R) - σ) < 1e-12 * norm(σ)
        end
    end

    @testset "component and tensor methods agree" begin
        for D in (2, 3)
            N = 30
            v = [SVector{D,Float64}(randn(D)) for _ in 1:N]
            energies = randn(N) ./ 40
            weights = rand(N) .+ 0.5
            T = 0.01
            L = randn(N, N) + N * I

            σ = conductivity(L, v, energies, weights, T)
            @test size(σ) == (D, D)
            for i in 1:D, j in 1:D
                @test conductivity(L, v, energies, weights, T, i, j) ≈ σ[i, j]
            end

            # `factorize` has no inferrable return type, so the correlator's element type is derived
            # from the operator instead; the static and shifted paths stay separately type-stable.
            @test (@inferred conductivity(L, v, energies, weights, T)) ≈ σ
            @test conductivity(L, v, energies, weights, T; ω = 0.1) isa SMatrix{D,D,ComplexF64}

            for tensor_of ∈ (thermal_conductivity, thermoelectric_conductivity, peltier_tensor)
                X = tensor_of(L, v, energies, weights, T)
                @test size(X) == (D, D)
                for i in 1:D, j in 1:D
                    @test tensor_of(L, v, energies, weights, T, i, j) ≈ X[i, j]
                end
            end

            # The thermal correlators are the shared primitive with energy currents substituted.
            weight = boltzmann_weight(energies, weights, T)
            @test thermal_conductivity(L, v, energies, weights, T, 1, 2) ≈
                  correlator(energies .* getindex.(v, 1), energies .* getindex.(v, 2), L, weight)
            @test thermoelectric_conductivity(L, v, energies, weights, T, 1, 2) ≈
                  correlator(getindex.(v, 1), energies .* getindex.(v, 2), L, weight)
            @test peltier_tensor(L, v, energies, weights, T, 1, 2) ≈
                  correlator(energies .* getindex.(v, 1), getindex.(v, 2), L, weight)
        end
    end

    @testset "generic effective lifetime" begin
        # ⟨a|L⁻¹|a⟩/⟨a|a⟩ = τ for L = I/τ whatever the driving vector, so one function covers both the
        # conductivity channel (a velocity component) and a viscous one (a deformation potential).
        N, τ = 40, 7.0
        v = [SVector{2,Float64}(randn(2)) for _ in 1:N]
        energies = randn(N) ./ 40
        weights = rand(N) .+ 0.5
        T = 0.01
        L = Matrix{Float64}(I / τ, N, N)

        @test effective_lifetime(L, getindex.(v, 1), energies, weights, T) ≈ hbar * τ
        @test effective_lifetime(L, randn(N), energies, weights, T) ≈ hbar * τ
    end

    @testset "argument checks" begin
        N, D = 12, 2
        v = [SVector{D,Float64}(randn(D)) for _ in 1:N]
        energies = randn(N) ./ 40
        weights = rand(N) .+ 0.5
        T = 0.01
        L = randn(N, N) + N * I

        # IBZ-length arrays against a full-BZ operator: the failure Auguste flagged as silent.
        @test_throws DimensionMismatch conductivity(L, v[1:6], energies[1:6], weights[1:6], T)
        @test_throws DimensionMismatch conductivity(L, v, energies[1:6], weights, T)
        @test_throws DimensionMismatch conductivity(L, v[1:6], energies, weights, T)
        @test_throws ArgumentError conductivity(L, v, energies, weights, T, 1, 3)
        @test_throws ArgumentError conductivity(L, v, energies, weights, T, 0, 1)
        # A shift needs the matrix itself, so a factorization cannot serve the AC/finite-q path.
        @test_throws ArgumentError conductivity(lu(L), v, energies, weights, T; ω = 0.1)
    end

end
