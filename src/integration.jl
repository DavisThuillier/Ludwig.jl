export populate_matrix!, impurity_scattering!

const η::SVector{6, Float64} = [1.0, 0.0, 1.0, 0.0, -1.0, 0.0]
const ρ::Float64 = 4*6^(1/3)/pi
const vol::Float64 = (8 * pi^2 / 15) # Volume of 5-dimensional unit sphere

# e-e scattering
function Γabc!(ζ::MVector{6, Float64}, a::Patch, b::Patch, c::Patch, T::Float64, Δε::Float64, h::Function)
    @inline εabc::Float64 = h(a.momentum + b.momentum - c.momentum) # Energy from momentum conservation
    δ::Float64 = a.energy + b.energy - c.energy - εabc # Energy violation

    v::SVector{2,Float64} = ForwardDiff.gradient(x -> h(x + b.momentum - c.momentum), a.momentum)
    
    # ζ[1], ζ[2] = v' * a.jinv
    # ζ[3], ζ[4] = v' * b.jinv
    # ζ[5], ζ[6] = - v' * c.jinv
    ζ[1] = v[1] * a.jinv[1,1] + v[2] * a.jinv[2,1]
    ζ[2] = v[1] * a.jinv[1,2] + v[2] * a.jinv[2,2]
    ζ[3] = v[1] * b.jinv[1,1] + v[2] * b.jinv[2,1]
    ζ[4] = v[1] * b.jinv[1,2] + v[2] * b.jinv[2,2]
    ζ[5] = - (v[1] * c.jinv[1,1] + v[2] * c.jinv[2,1])
    ζ[6] = - (v[1] * c.jinv[1,2] + v[2] * c.jinv[2,2])
    u::SVector{6, Float64} = (Δε / 2) * η - ζ

    if abs(δ) < Δε * 1e-4
        return vol * a.djinv * b.djinv * c.djinv * ρ^(5/2) * (1 - f0(εabc, T)) / norm(u)
    else
        # εabc ≈ ε0 + ζ . x
        ρ < δ^2 / dot(u,u) && return 0.0 # Check for intersection of energy conserving 5-plane with coordinate space

        xpara::SVector{6, Float64} = - δ * u / dot(u,u) # Linearized coordinate along energy conserving direction

        r5::Float64 = (ρ - δ^2 / dot(u,u) )^(5/2)

        return vol * a.djinv * b.djinv * c.djinv * r5 * (1 - f0(εabc + dot(ζ, xpara), T)) / norm(u)
    end
end

function Γabc!(ζ::MVector{6, Float64}, a::Patch, b::Patch, c::Patch, T::Float64, Δε::Float64, itp::ScaledInterpolation)
    d::MVector{2,Float64} = a.momentum + b.momentum - c.momentum
    d[1] = mod(d[1] + 0.5, 1.0) - 0.5
    d[2] = mod(d[2] + 0.5, 1.0) - 0.5
    @inbounds εabc::Float64 = itp(d[1], d[2]) # Energy from momentum conservation
    δ::Float64 = a.energy + b.energy - c.energy - εabc # Energy violatio

    @inbounds v::SVector{2,Float64} = Interpolations.gradient(itp, d[1], d[2])
    
    # ζ[1], ζ[2] = v' * a.jinv
    # ζ[3], ζ[4] = v' * b.jinv
    # ζ[5], ζ[6] = - v' * c.jinv
    ζ[1] = v[1] * a.jinv[1,1] + v[2] * a.jinv[2,1]
    ζ[2] = v[1] * a.jinv[1,2] + v[2] * a.jinv[2,2]
    ζ[3] = v[1] * b.jinv[1,1] + v[2] * b.jinv[2,1]
    ζ[4] = v[1] * b.jinv[1,2] + v[2] * b.jinv[2,2]
    ζ[5] = - (v[1] * c.jinv[1,1] + v[2] * c.jinv[2,1])
    ζ[6] = - (v[1] * c.jinv[1,2] + v[2] * c.jinv[2,2])
    u::SVector{6, Float64} = (Δε / 2) * η - ζ
    # u::SVector{6, Float64} = [a.de, b.de, c.de] - ζ

    if abs(δ) < Δε * 1e-4
        return vol * a.djinv * b.djinv * c.djinv * ρ^(5/2) * (1 - f0(εabc, T)) / norm(u)
    else
        # εabc ≈ ε0 + ζ . x
        ρ < δ^2 / dot(u,u) && return 0.0 # Check for intersection of energy conserving 5-plane with coordinate space

        xpara::SVector{6, Float64} = - δ * u / dot(u,u) # Linearized coordinate along energy conserving direction

        r5::Float64 = (ρ - δ^2 / dot(u,u) )^(5/2)

        return vol * a.djinv * b.djinv * c.djinv * r5 * (1 - f0(εabc + dot(ζ, xpara), T)) / norm(u)
    end
end

quadrant(k::SVector{2,Float64}) = mod(ceil(2 * atan(k[2], k[1]) / pi) + 3, 4)

# Impurity scattering for Sr2RuO4
function Iab(a::Patch, b::Patch, Δε::Float64, band::String)
    if abs(a.energy - b.energy) < Δε/2 
        if band == "alpha"
            qa = quadrant(a.momentum)
            pa = norm(a.momentum - [0.5 * (-1)^floor((qa + 1) / 2), 0.5*(-1)^floor(qa/2)])
            qb = quadrant(b.momentum)
            pb = norm(b.momentum - [0.5 * (-1)^floor((qb + 1) / 2), 0.5*(-1)^floor(qb/2)])
        else
            pa = norm(a.momentum)
            pb = norm(b.momentum)
        end
        # V = 2 * ((norm(a.v)/pa)^2 + (norm(b.v)/pb)^2)
        V = 4 * sqrt((norm(a.v)/pa)^2 * (norm(b.v)/pb)^2) # really, V^2
        return 16 * V * a.djinv * b.djinv / Δε 
    else
        return 0.0
    end
end

function Iab(a::Patch, b::Patch, Δε::Float64, V_squared::Function)
    if abs(a.energy - b.energy) < Δε/2 
        return 16 * V_squared(a.momentum, b.momentum) * a.djinv * b.djinv / Δε 
    else
        return 0.0
    end
end

# Simple impurity Scattering with no momentum dependence
function Iab(a::Patch, b::Patch, Δε::Float64)
    if abs(a.energy - b.energy) < Δε/2 
        V = ((norm(a.v)/norm(a.momentum))^2 + (norm(b.v)/norm(b.momentum))^2)/ (2 * pi^2)
        return 16 * V * a.djinv * b.djinv / Δε 
    else
        return 0.0
    end
end

"""
    populate_matrix!(Γ, grid, Δε, T, h)

Compute the linearized Boltzmann collision operator at finite temperature `T` sampled on `grid` with Hamiltonian `h` for both electron-electron and impurity scattering. 

Assumptions:
`h` is a tight-binding Hamiltonian, and `T`, `Δε` are in units of the nearest-neighbor hopping parameter t. It is assumed the final matrix will be scaled by |U|^2 where U is also in units of t.
"""
function populate_matrix!(Γ::Matrix{Float64}, grid::Vector{Patch}, Δε::Float64, T::Float64, h::Function)
    f0s  = map(x -> f0(x.energy, T), grid)

    ζ = MVector{6, Float64}(undef)
    Iibc = zeros(Float64, length(grid), length(grid))
    for i in ProgressBar(eachindex(grid))
        for j in eachindex(grid)
            @inbounds Iibc[j, :] .= Γabc!.(Ref(ζ), Ref(grid[i]), Ref(grid[j]), grid, Ref(T), Ref(Δε), Ref(h))
        end
        
        for j in eachindex(grid)
            i == j && continue
            @inbounds Γ[i,j] += dot(Iibc[j,:], (1 .- f0s)) * f0s[j] / (1 - f0(grid[i].energy, T)) 
            @inbounds Γ[i,j] -= 2 * dot(Iibc[:,j], f0s) * (1 - f0s[j]) / (1 - f0(grid[i].energy, T))
        end

        Γ[i, :] /= grid[i].dV
    end
end

function populate_matrix!(Γ::Matrix{Float64}, grid::Vector{Patch}, Δε::Float64, T::Float64, itp::ScaledInterpolation)
    f0s  = map(x -> f0(x.energy, T), grid)

    ζ = MVector{6, Float64}(undef)
    Iibc = zeros(Float64, length(grid), length(grid))
    for i in ProgressBar(eachindex(grid))
        for j in eachindex(grid)
            @inbounds Iibc[j, :] .= Γabc!.(Ref(ζ), Ref(grid[i]), Ref(grid[j]), grid, Ref(T), Ref(Δε), Ref(itp))
        end
        
        for j in eachindex(grid)
            i == j && continue
            @inbounds Γ[i,j] += dot(Iibc[j,:], (1 .- f0s)) * f0s[j] / (1 - f0(grid[i].energy, T)) 
            @inbounds Γ[i,j] -= 2 * dot(Iibc[:,j], f0s) * (1 - f0s[j]) / (1 - f0(grid[i].energy, T))
        end

        Γ[i, :] /= grid[i].dV
    end
end

# function populate_matrix!(Γ::Matrix{Float64}, grid::Vector{Patch}, Δε::Float64, T::Float64, itp::ScaledInterpolation)
#     f0s  = map(x -> f0(x.energy, T), grid)

#     ζ = MVector{6, Float64}(undef)
#     Threads.@threads for i in ProgressBar(eachindex(grid))
#         for j in eachindex(grid)
#             i == j && continue
#             acc::Float64 = 0.0
#             for k in eachindex(grid)
#                 @inbounds acc += Γabc!(ζ, grid[i], grid[j], grid[k], T, Δε, itp) * (1 - f0s[i]) * f0s[j] / (1 - f0(grid[i].energy, T))
#                 @inbounds acc -= 2 * Γabc!(ζ, grid[i], grid[k], grid[j], T, Δε, itp) * (1 - f0s[j]) / (1 - f0(grid[i].energy, T))
#             end
#             @inbounds Γ[i,j] = acc
#         end

#         Γ[i, :] /= grid[i].dV
#     end
# end


function impurity_scattering!(Γ::Matrix{Float64}, grid::Vector{Patch}, Δε::Float64, band::String)
    for i in ProgressBar(eachindex(grid))        
        for j in eachindex(grid)
            i == j && continue
            @inbounds Γ[i,j] -= Iab(grid[i], grid[j], Δε, band)
        end

        Γ[i, :] /= grid[i].dV
    end
    return nothing
end

function impurity_scattering!(Γ::Matrix{Float64}, grid::Vector{Patch}, Δε::Float64, V_squared::Function)
    for i in ProgressBar(eachindex(grid))        
        for j in eachindex(grid)
            i == j && continue
            @inbounds Γ[i,j] -= Iab(grid[i], grid[j], Δε, V_squared)
        end

        Γ[i, :] /= grid[i].dV
    end
    return nothing
end