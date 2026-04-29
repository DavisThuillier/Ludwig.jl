using LinearAlgebra
using Random

###
### Shared workload
###

const T_prop     = 0.005
const Δε_prop    = 1.5 * T_prop
const n_arc_prop = 6
const N_grid_prop = 401
const lat_prop   = Lattice([1.0 0.0; 0.0 1.0])
ε_prop(k)        = -2.0 * (cos(k[1]) + cos(k[2]))

const mesh_prop = ibz_mesh(lat_prop, [ε_prop], T_prop, Δε_prop, n_arc_prop, N_grid_prop)
const grid_prop = patches(mesh_prop)
const N_prop    = length(grid_prop)
const v_prop    = [p.v for p in grid_prop]
const E_prop    = [p.e for p in grid_prop]
const dV_prop   = [p.dV for p in grid_prop]

# Synthetic symmetric, diagonally-dominant collision matrix. Real-symmetric so the
# transport tensors stay symmetric; positive-definite so lifetimes come out positive.
function build_synthetic_L(N, seed)
    rng    = MersenneTwister(seed)
    A_off  = 0.05 .* randn(rng, N, N)
    A_off  = (A_off .+ A_off') ./ 2
    for i in 1:N
        A_off[i, i] = 0.0
    end
    return Matrix(Diagonal(1.0 .+ 0.1 .* rand(rng, N))) .+ A_off
end

const L_prop = build_synthetic_L(N_prop, 0x1d775c8c)

@testset "inner_product" begin
    rng = MersenneTwister(0xc0ffee)
    a   = randn(rng, N_prop)
    b   = randn(rng, N_prop)
    w   = ones(N_prop)

    ip = inner_product(a, b, L_prop, w)
    @test isa(ip, Number)
    @test isfinite(ip)

    ip_aa = inner_product(a, a, L_prop, w)
    @test abs(imag(ip_aa)) < 1e-10

    F    = lu(L_prop)
    ip_F = inner_product(a, b, F, w)
    @test isapprox(ip, ip_F; atol = 1e-8)
end

@testset "electrical_conductivity" begin
    σ = electrical_conductivity(L_prop, v_prop, E_prop, dV_prop, T_prop)
    @test isa(σ, Matrix{ComplexF64})
    @test size(σ) == (2, 2)
    @test all(isfinite, σ)
    @test all(abs.(imag.(σ)) .< 1e-10)

    σω = electrical_conductivity(L_prop, v_prop, E_prop, dV_prop, T_prop, 1.0)
    @test isa(σω, Matrix{ComplexF64})
    @test all(isfinite, σω)

    σq = electrical_conductivity(
        L_prop, v_prop, E_prop, dV_prop, T_prop, 0.0, [0.1, 0.0]
    )
    @test isa(σq, Matrix{ComplexF64})
    @test all(isfinite, σq)
end

@testset "longitudinal_electrical_conductivity" begin
    vx  = first.(v_prop)
    σxx = longitudinal_electrical_conductivity(L_prop, vx, E_prop, dV_prop, T_prop)
    @test isa(σxx, Number)
    @test isfinite(σxx)
    @test abs(imag(σxx)) < 1e-10

    σxx_ω = longitudinal_electrical_conductivity(
        L_prop, vx, E_prop, dV_prop, T_prop, 1.0
    )
    @test isfinite(σxx_ω)
end

@testset "thermal_conductivity" begin
    κ = thermal_conductivity(L_prop, v_prop, E_prop, dV_prop, T_prop)
    @test isa(κ, Matrix{ComplexF64})
    @test size(κ) == (2, 2)
    @test all(isfinite, κ)

    κ11 = thermal_conductivity(L_prop, v_prop, E_prop, dV_prop, T_prop, 1, 1)
    @test isa(κ11, Number)
    @test isfinite(κ11)
    @test isapprox(κ11, κ[1, 1]; atol = 1e-8)

    @test_throws ArgumentError thermal_conductivity(
        L_prop, v_prop, E_prop, dV_prop, T_prop, 0, 1
    )
    @test_throws ArgumentError thermal_conductivity(
        L_prop, v_prop, E_prop, dV_prop, T_prop, 1, 3
    )
end

@testset "thermoelectric_conductivity" begin
    ϵ = thermoelectric_conductivity(L_prop, v_prop, E_prop, dV_prop, T_prop)
    @test isa(ϵ, Matrix{ComplexF64})
    @test size(ϵ) == (2, 2)
    @test all(isfinite, ϵ)

    ϵ11 = thermoelectric_conductivity(L_prop, v_prop, E_prop, dV_prop, T_prop, 1, 1)
    @test isa(ϵ11, Number)
    @test isfinite(ϵ11)
    @test isapprox(ϵ11, ϵ[1, 1]; atol = 1e-8)

    @test_throws ArgumentError thermoelectric_conductivity(
        L_prop, v_prop, E_prop, dV_prop, T_prop, 0, 1
    )
    @test_throws ArgumentError thermoelectric_conductivity(
        L_prop, v_prop, E_prop, dV_prop, T_prop, 1, 3
    )
end

@testset "peltier_tensor" begin
    τ = peltier_tensor(L_prop, v_prop, E_prop, dV_prop, T_prop)
    @test isa(τ, Matrix{ComplexF64})
    @test size(τ) == (2, 2)
    @test all(isfinite, τ)

    τ11 = peltier_tensor(L_prop, v_prop, E_prop, dV_prop, T_prop, 1, 1)
    @test isa(τ11, Number)
    @test isfinite(τ11)
    @test isapprox(τ11, τ[1, 1]; atol = 1e-8)

    @test_throws ArgumentError peltier_tensor(
        L_prop, v_prop, E_prop, dV_prop, T_prop, 0, 1
    )
    @test_throws ArgumentError peltier_tensor(
        L_prop, v_prop, E_prop, dV_prop, T_prop, 1, 3
    )
end

@testset "ηB1g and ηB2g" begin
    Dxx = first.(getfield.(grid_prop, :k))
    Dyy = last.(getfield.(grid_prop, :k))
    Dxy = Dxx .* Dyy

    η1 = ηB1g(L_prop, E_prop, dV_prop, Dxx, Dyy, T_prop)
    @test isa(η1, Number)
    @test isfinite(η1)
    @test η1 != 0

    η2 = ηB2g(L_prop, E_prop, dV_prop, Dxy, T_prop)
    @test isa(η2, Number)
    @test isfinite(η2)
    @test η2 != 0
end

@testset "lifetimes" begin
    τσ = σ_lifetime(L_prop, v_prop, E_prop, dV_prop, T_prop)
    @test isa(τσ, Float64)
    @test isfinite(τσ)
    @test τσ > 0

    D  = first.(getfield.(grid_prop, :k))
    τη = η_lifetime(L_prop, D, E_prop, dV_prop, T_prop)
    @test isfinite(τη)
    @test real(τη) > 0
end
