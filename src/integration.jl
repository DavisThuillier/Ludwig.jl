export populate_matrix!, impurity_scattering!

const η::SVector{6, Float64} = [1.0, 0.0, 1.0, 0.0, -1.0, 0.0]
const ρ::Float64 = 4*6^(1/3)/pi
const vol::Float64 = (8 * pi^2 / 15) # Volume of 5-dimensional unit sphere

map_to_first_bz(k) = SVector(mod.(k .+ 0.5, 1.0) .- 0.5)

# e-e scattering
function Γabc!(ζ::MVector{6, Float64}, a::Patch, b::Patch, c::Patch, T::Real, Δε::Real, hams::Vector{Function})
    d::MVector{2,Float64} = a.momentum + b.momentum - c.momentum
    d[1] = mod(d[1] + 0.5, 1.0) - 0.5
    d[2] = mod(d[2] + 0.5, 1.0) - 0.5

    integral = Vector{Float64}(undef, length(itps))
    for (i, h) in enumerate(hams)
        @inline εabc = h(d) # Energy from momentum conservation
        δ = a.energy + b.energy - c.energy - εabc # Energy conservation violations

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
            integral[i] = ρ^(5/2) * (1 - f0(εabc, T)) / norm(u)
        else
            # εabc ≈ ε0 + ζ . x
            ρ < δ^2 / dot(u,u) && continue # Check for intersection of energy conserving 5-plane with coordinate space

            xpara::SVector{6, Float64} = - δ * u / dot(u,u) # Linearized coordinate along energy conserving direction

            r5::Float64 = (ρ - δ^2 / dot(u,u) )^(5/2)

            integral[i] = r5 * (1 - f0(εabc + dot(ζ, xpara), T)) / norm(u)
        end
    end
    return vol * a.djinv * b.djinv * c.djinv * integral
end

Γabc!(ζ::MVector{6, Float64}, a::Patch, b::Patch, c::Patch, T::Float64, Δε::Real, h::Function) = Γabc!(ζ, a, b, c, T, Δε, [h])

function Γabc!(ζ::MVector{6, Float64}, a::Patch, b::Patch, c::Patch, T::Float64, Δε::Real, itps::Vector{ScaledInterpolation})
    d::MVector{2,Float64} = a.momentum + b.momentum - c.momentum
    d[1] = mod(d[1] + 0.5, 1.0) - 0.5
    d[2] = mod(d[2] + 0.5, 1.0) - 0.5

    integral = SVector{length(itps), Float64}(undef)
    for (i, itp) in enumerate(itps)
        @inbounds εabc = itp(d[1], d[2])  # Energies from momentum conservation
        δ = a.energy + b.energy - c.energy - εabc # Energy conservation violations

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
        if abs(δ) < Δε * 1e-4
            integral[i] = ρ^(5/2) * (1 - f0(εabc, T)) / norm(u)
        else
            # εabc ≈ ε0 + ζ . x
            ρ < δ^2 / dot(u,u) && continue # Check for intersection of energy conserving 5-plane with coordinate space

            xpara::SVector{6, Float64} = - δ * u / dot(u,u) # Linearized coordinate along energy conserving direction

            r5::Float64 = (ρ - δ^2 / dot(u,u) )^(5/2)

            integral[i] = r5 * (1 - f0(εabc + dot(ζ, xpara), T)) / norm(u)
        end
    end
    return vol * a.djinv * b.djinv * c.djinv * integral
end

Γabc!(ζ::MVector{6, Float64}, a::Patch, b::Patch, c::Patch, T::Float64, Δε::Real, itp::ScaledInterpolation) = Γabc!(ζ, a, b, c, T, Δε, [itp])

function Iab(a::Patch, b::Patch, Δε::Float64, V_squared::Function)
    if abs(a.energy - b.energy) < Δε/2 
        return 16 * V_squared(a.momentum, b.momentum) * a.djinv * b.djinv / Δε 
    else
        return 0.0
    end
end

function electron_electron!(Γ::AbstractArray{<:Real,2}, grid::Vector{Patch}, Δε::Real, T::Real, hams::Vector{Function})
    f0s  = map(x -> f0(x.energy, T), grid) # Fermi-Dirac Grid

    ζ = MVector{6, Float64}(undef)
    Iibc = zeros(Float64, length(grid), length(grid))
    for i in ProgressBar(eachindex(grid))
        for j in eachindex(grid)
            @inbounds Iibc[j, :] .= sum.(Γabc!.(Ref(ζ), Ref(grid[i]), Ref(grid[j]), grid, Ref(T), Ref(Δε), Ref(hams)))
        end
        
        for j in eachindex(grid)
            i == j && continue
            @inbounds Γ[i,j] += dot(Iibc[j,:], (1 .- f0s)) * f0s[j] / (1 - f0(grid[i].energy, T)) 
            @inbounds Γ[i,j] -= 2 * dot(Iibc[:,j], f0s) * (1 - f0s[j]) / (1 - f0(grid[i].energy, T))
        end

        Γ[i, :] /= grid[i].dV
    end
end

function electron_electron!(Γ::AbstractArray{<:Real,2}, grid::Vector{Patch}, Δε::Real, T::Real, itps::Vector{ScaledInterpolation})
    f0s  = map(x -> f0(x.energy, T), grid)

    ζ = MVector{6, Float64}(undef)
    Iibc = zeros(Float64, length(grid), length(grid))
    for i in ProgressBar(eachindex(grid))
        for j in eachindex(grid)
            @inbounds Iibc[j, :] .= sum.(Γabc!.(Ref(ζ), Ref(grid[i]), Ref(grid[j]), grid, Ref(T), Ref(Δε), Ref(itps)))
        end
        
        for j in eachindex(grid)
            i == j && continue
            @inbounds Γ[i,j] += dot(Iibc[j,:], (1 .- f0s)) * f0s[j] / (1 - f0(grid[i].energy, T)) 
            @inbounds Γ[i,j] -= 2 * dot(Iibc[:,j], f0s) * (1 - f0s[j]) / (1 - f0(grid[i].energy, T))
        end

        Γ[i, :] /= grid[i].dV
    end
end

function electron_electron!(Γ::AbstractArray{<:Real,2}, grid::Vector{Patch}, Δε::Real, T::Real, H::Function, N::Int)
    itps = Ludwig.get_bands(H, N)

    f0s  = map(x -> f0(x.energy, T), grid) # Fermi-Dirac Grid

    ζ = MVector{6, Float64}(undef)
    Ki = Matrix{SVector{3, Float64}}(undef, length(grid), length(grid))
    for i in ProgressBar(eachindex(grid))
        Pi = grid[i] 

        for j in eachindex(grid)
            @inbounds Ki[j, :] .= Γabc!.(Ref(ζ), Ref(grid[i]), Ref(grid[j]), grid, Ref(T), Ref(Δε), Ref(itps))
        end
        
        for j in eachindex(grid)
            i == j && continue
            Pj = grid[j]

            Fji = dot(Pj.w, Pi.w)
            for m in eachindex(grid)
                Pm = grid[m]

                Fmi = dot(Pm.w, Pi.w)
                if Fmi != 0
                    k1 = map_to_first_bz(Pi.momentum + Pj.momentum - Pm.momentum)
                    W1 = eigvecs(H(k1))

                    
                end


                
                k2 = map_to_first_bz(Pi.momentum + Pm.momentum - Pj.momentum)
                



                # Depopulation ##
                W2 = eigvecs(H(k2))

            end

            # @inbounds Γ[i,j] += dot(Iibc[j,:], (1 .- f0s)) * f0s[j] / (1 - f0(grid[i].energy, T)) 
            # @inbounds Γ[i,j] -= 2 * dot(Iibc[:,j], f0s) * (1 - f0s[j]) / (1 - f0(grid[i].energy, T))
        end

        Γ[i, :] /= grid[i].dV
    end
end

function electron_impurity!(Γ::AbstractArray{<:Real,2}, grid::Vector{Patch}, Δε::Real, V_squared)
    for i in ProgressBar(eachindex(grid))        
        for j in eachindex(grid)
            i == j && continue
            @inbounds Γ[i,j] -= Iab(grid[i], grid[j], Δε, V_squared)
        end

        Γ[i, :] /= grid[i].dV
    end
    return nothing
end

function electron_impurity!(Γ::AbstractArray{<:Real,2}, grid::Vector{Patch}, Δε::Real, V_squared::Matrix)
    ℓ = size(Γ)[1] ÷ 3
    for i in ProgressBar(eachindex(grid))        
        for j in eachindex(grid)
            i == j && continue
            @inbounds Γ[i,j] -= Iab(grid[i], grid[j], Δε, (x,y) -> V_squared[(i-1)÷ℓ + 1, (j-1)÷ℓ + 1])
        end

        Γ[i, :] /= grid[i].dV
    end
    return nothing
end