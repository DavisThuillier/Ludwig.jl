export Γabc!, electron_electron

const ρ::Float64 = 4*6^(1/3)/pi

"Volume of the 5-dimensional unit sphere"
const vol::Float64 = (8 * pi^2 / 15)

"""
    Kabc!(ζ, u, a::Patch, b::Patch, c::Patch, T, εabc, ε::Function)

Compute the integral
```math
    \\mathcal{K}_{abc} \\equiv \\int_a \\int_b \\int_c (1 - f^{(0)}(\\mathbf{k}_a + \\mathbf{k}_b + \\mathbf{k}_c)) \\delta(\\varepsilon_a + \\varepsilon_b - \\varepsilon_c - \\varepsilon(\\mathbf{k}_a + \\mathbf{k}_b - \\mathbf{k}_c))
```
with dispersion `ε` at temperature `T`.
```math
    \\int_i \\equiv \\frac{1}{a^2} \\int_{\\mathbf{k} \\in \\mathcal{P}_i} d^2\\mathbf{k}
```
is an integral over momenta in patch ``\\mathcal{P}_i``.

"""
function Kabc!(ζ, u, a::Patch, b::Patch, c::Patch, T::Real, εabc, ε::Function)
    δ = a.energy + b.energy - c.energy - εabc # Energy conservation violations

    v::SVector{2,Float64} = ForwardDiff.gradient(x -> ε(x + b.momentum - c.momentum), a.momentum)

    # ζ[1], ζ[2] = v' * a.jinv
    # ζ[3], ζ[4] = v' * b.jinv
    # ζ[5], ζ[6] = - v' * c.jinv
    ζ[1] = v[1] * a.jinv[1,1] + v[2] * a.jinv[2,1]
    ζ[2] = v[1] * a.jinv[1,2] + v[2] * a.jinv[2,2]
    ζ[3] = v[1] * b.jinv[1,1] + v[2] * b.jinv[2,1]
    ζ[4] = v[1] * b.jinv[1,2] + v[2] * b.jinv[2,2]
    ζ[5] = - (v[1] * c.jinv[1,1] + v[2] * c.jinv[2,1])
    ζ[6] = - (v[1] * c.jinv[1,2] + v[2] * c.jinv[2,2])

    # u = [a.de/2.0, 0.0, b.de/2.0, 0.0, -c.de/2.0, 0.0] - ζ
    u[1] = a.de/2.0 - ζ[1]
    u[2] = - ζ[2]
    u[3] = b.de/2.0 - ζ[3]
    u[4] = - ζ[4]
    u[5] = (- c.de/2.0) - ζ[5]
    u[6] = - ζ[6]


    # εabc ≈ ε0 + ζ . x
    ρ < δ^2 / dot(u,u) && return 0.0 # Check for intersection of energy conserving 5-plane with coordinate space

    Δε = - δ * dot(ζ, u) / dot(u,u)
    r5 = (ρ - δ^2 / dot(u,u) )^(5/2)

    return vol * a.djinv * b.djinv * c.djinv * r5 * (1 - f0(εabc + Δε, T)) / norm(u)

end

"""
    Kabc!(ζ, u, a::Patch, b::Patch, c::Patch, T, k, εabc, itp)

Compute ``\\mathcal{K}_{abc} `` with `k` given by momentum conservation using the representative central momenta of patches `a`, `b`, and `c`, using `itp` as an interpolation of the dispersion.

"""
function Kabc!(ζ, u, a::Patch, b::Patch, c::Patch, T::Real, k, εabc, itp::ScaledInterpolation)
    δ = a.energy + b.energy - c.energy - εabc # Energy conservation violations

    @inbounds v::SVector{2,Float64} = Interpolations.gradient(itp, k[1], k[2])

    # ζ[1], ζ[2] = v' * a.jinv
    # ζ[3], ζ[4] = v' * b.jinv
    # ζ[5], ζ[6] = - v' * c.jinv
    ζ[1] = v[1] * a.jinv[1,1] + v[2] * a.jinv[2,1]
    ζ[2] = v[1] * a.jinv[1,2] + v[2] * a.jinv[2,2]
    ζ[3] = v[1] * b.jinv[1,1] + v[2] * b.jinv[2,1]
    ζ[4] = v[1] * b.jinv[1,2] + v[2] * b.jinv[2,2]
    ζ[5] = - (v[1] * c.jinv[1,1] + v[2] * c.jinv[2,1])
    ζ[6] = - (v[1] * c.jinv[1,2] + v[2] * c.jinv[2,2])

    # u = [a.de/2.0, 0.0, b.de/2.0, 0.0, -c.de/2.0, 0.0] - ζ
    u[1] = a.de/2.0 - ζ[1]
    u[2] = - ζ[2]
    u[3] = b.de/2.0 - ζ[3]
    u[4] = - ζ[4]
    u[5] = (- c.de/2.0) - ζ[5]
    u[6] = - ζ[6]

    ρ < δ^2 / dot(u,u) && return 0.0 # Check for intersection of energy conserving 5-plane with coordinate space

    Δε = - δ * dot(ζ, u) / dot(u,u)
    r5 = (ρ - δ^2 / dot(u,u) )^(5/2)

    return vol * a.djinv * b.djinv * c.djinv * r5 * (1 - f0(εabc + Δε, T)) / norm(u)
end

"""
    electron_electron(grid::Vector{Patch}, f0s::Vector{Float64}, i::Int, j::Int, itps::Vector{ScaledInterpolation}, T::Real, Fpp::Function, Fpk::Function)

Compute the element (`i`,`j`) of the linearized Boltzmann collision operator for electron electron scattering.

The bands used to construct `grid` are callable using the interpolated dispersion in `itps`. The vector `f0s` stores the value of the Fermi-Dirac distribution at each patch center an can be calculated independent of `i` and `j`. The functions `Fpp` and `Fpk` are vertex factors defined for two patch variables and for one patch and one momentum vector respectively.

"""
function electron_electron(grid::Vector{Patch}, f0s::Vector{Float64}, i::Int, j::Int, itps::Vector{ScaledInterpolation}, T::Real, Fpp::Function, Fpk::Function, Ω)
    Lij::Float64 = 0.0
    w123::Float64 = 0.0
    w124::Float64 = 0.0

    cutoff = - Ω * T

    ζ  = MVector{6,Float64}(undef)
    u  = MVector{6,Float64}(undef)

    kij = grid[i].momentum + grid[j].momentum
    qij = grid[i].momentum - grid[j].momentum
    kijm = Vector{Float64}(undef, 2)
    qimj = Vector{Float64}(undef, 2)

    energies = Vector{Float64}(undef, length(itps))

    μ4::Int = 0
    μ34::Int = 0
    min_e::Float64 = 0

    for m in eachindex(grid)
        kijm .= mod.(kij .- grid[m].momentum .+ 0.5, 1.0) .- 0.5

        min_e = 1e3
        μ4 = 1
        for μ in eachindex(itps)
            energies[μ] = itps[μ](kijm[1], kijm[2])
            if abs(energies[μ]) < min_e
                min_e = abs(energies[μ]); μ4 = μ
            end
        end
        
        if energies[μ4] > cutoff

            w123 = Weff_squared_123(grid[i], grid[j], grid[m], Fpp, Fpk, kijm, μ4)

            if w123 != 0
                Lij += w123 * Kabc!(ζ, u, grid[i], grid[j], grid[m], T, kijm, energies[μ4], itps[μ4]) * f0s[j] * (1 - f0s[m])
            end

        end

        qimj .= mod.(qij .+ grid[m].momentum .+ 0.5, 1.0) .- 0.5

        min_e = 1e3
        μ34 = 1
        for μ in eachindex(itps)
            energies[μ] = itps[μ](qimj[1], qimj[2])
            if abs(energies[μ]) < min_e
                min_e = abs(energies[μ]); μ34 = μ
            end
        end

        if energies[μ4] > cutoff

            w123 = Weff_squared_123(grid[i], grid[m], grid[j], Fpp, Fpk, qimj, μ34)
            w124 = Weff_squared_124(grid[i], grid[m], grid[j], Fpp, Fpk, qimj, μ34)

            if w123 + w124 != 0
                Lij -= (w123 + w124) * Kabc!(ζ, u, grid[i], grid[m], grid[j], T, qimj, energies[μ34], itps[μ34]) * f0s[m] * (1 - f0s[j])
            end

        end

    end

    return Lij / (grid[i].dV * (1 - f0s[i]))
end

"""
    Iab(a, b, V_squared)

Compute the scattering amplitude for an electron scattering from patch `a` to patch `b` with `V_squared` given by
```math
    | \\langle \\mathbf{k}_a | \\hat{V} | \\mathbf{k}_b \\rangle |^2 = |V(\\mathbf{k}_a, \\mathbf{k}_b)|^2.
```
"""
function Iab(a::Patch, b::Patch, V_squared::Function)
    Δε = sqrt(a.de^2 + b.de^2) / sqrt(2) # Add energy differentials in quadrature to obtain ||u||
    if abs(a.energy - b.energy) < Δε/2
        return 16 * V_squared(a.momentum, b.momentum) * a.djinv * b.djinv / Δε
    else
        return 0.0
    end
end

"""
    electron_impurity
"""
function electron_impurity!(L::AbstractArray{<:Real,2}, grid::Vector{Patch}, V_squared)
    for i in eachindex(grid)
        for j in eachindex(grid)
            i == j && continue
            @inbounds L[i,j] -= Iab(grid[i], grid[j], V_squared)
        end

        L[i, :] /= grid[i].dV
    end
    return nothing
end

function electron_impurity!(L::AbstractArray{<:Real,2}, grid::Vector{Patch}, V_squared::Matrix)
    ℓ = size(L)[1] ÷ size(V_squared)[1]
    for i in eachindex(grid)
        for j in eachindex(grid)
            i == j && continue
            @inbounds L[i,j] -= Iab(grid[i], grid[j], (x,y) -> V_squared[(i-1)÷ℓ + 1, (j-1)÷ℓ + 1])
        end

        L[i, :] /= grid[i].dV
    end
    return nothing
end