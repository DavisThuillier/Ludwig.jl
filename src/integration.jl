# const η::SVector{6, Float64} = [1.0, 0.0, 1.0, 0.0, -1.0, 0.0]
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
    d = map_to_first_bz(a.momentum + b.momentum - c.momentum)
    integral = MVector{length(itps), Float64}(undef)

    δ::Float64 = 0.0 # Preallocate

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

            r5 = (ρ - δ^2 / dot(u,u) )^(5/2)

            integral[i] = r5 * (1 - f0(εabc + dot(ζ, xpara), T)) / norm(u)
        end
    end

    return vol * a.djinv * b.djinv * c.djinv * integral
end

Γabc!(ζ::MVector{6, Float64}, a::Patch, b::Patch, c::Patch, T::Float64, Δε::Real, itp::ScaledInterpolation) = Γabc!(ζ, a, b, c, T, Δε, [itp])

function ee_kernel!(K, grid::Vector{Patch}, i::Int, T, Δε, itps, f0s)
    # Preallocation
    k12 = MVector{2,Float64}(undef)
    k4 = MVector{2,Float64}(undef)
    v  = MVector{2,Float64}(undef)
    ζ  = MVector{6,Float64}(undef)
    u  = MVector{6,Float64}(undef)
    xpara  = MVector{6,Float64}(undef)
    
    δij::Float64 = 0.0
    δijm::Float64 = 0.0
    δ::Float64 = 0.0 

    εijm::Float64 = 0.0
    r5::Float64 = 0.0
    η::SVector{6, Float64} = (Δε / 2.0) * [1.0, 0.0, 1.0, 0.0, -1.0, 0.0]

    a = grid[i] # Reference patch

    for j in eachindex(grid)

        δij = a.energy + grid[j].energy
        k12 .= a.momentum .+ grid[j].momentum
        for m in eachindex(grid)
            δijm = δij - grid[m].energy

            k4 .= map_to_first_bz(k12 .- grid[m].momentum)
            for (μ, itp) in enumerate(itps)
                @inbounds εijm = itp(k4[1], k4[2])  # Energies from momentum conservation
                δ = δijm - εijm # Energy conservation violations
                v .= Interpolations.gradient(itp, k4[1], k4[2])
                
                # ζ[1], ζ[2] = v' * a.jinv
                # ζ[3], ζ[4] = v' * grid[j].jinv
                # ζ[5], ζ[6] = - v' * grid[m].jinv
                ζ[1] = v[1] * a.jinv[1,1] + v[2] * a.jinv[2,1]
                ζ[2] = v[1] * a.jinv[1,2] + v[2] * a.jinv[2,2]
                ζ[3] = v[1] * grid[j].jinv[1,1] + v[2] * grid[j].jinv[2,1]
                ζ[4] = v[1] * grid[j].jinv[1,2] + v[2] * grid[j].jinv[2,2]
                ζ[5] = - (v[1] * grid[m].jinv[1,1] + v[2] * grid[m].jinv[2,1])
                ζ[6] = - (v[1] * grid[m].jinv[1,2] + v[2] * grid[m].jinv[2,2])
        
                u .= η .- ζ
        
                if abs(δ) < Δε * 1e-4
                    @inbounds K[j,m,μ] = f0s[j] * (1 - f0s[m]) * vol * a.djinv * grid[j].djinv * grid[m].djinv * ρ^(5/2) * (1 - f0(εijm, T)) / norm(u)
                else
                    # εijm ≈ ε0 + ζ . x
                    if ρ < δ^2 / dot(u,u)
                        @inbounds (K[j,m,μ] = 0.0) # Check for intersection of energy conserving 5-plane with coordinate space
                    else
                        xpara .= (- δ / dot(u,u)) * u  # Linearized coordinate along energy conserving direction
            
                        r5 = (ρ - δ^2 / dot(u,u) )^(5/2)
            
                        @inbounds K[j,m,μ] = f0s[j] * (1 - f0s[m]) * vol * a.djinv * grid[j].djinv * grid[m].djinv * r5 * (1 - f0(εijm + dot(ζ, xpara), T)) / norm(u)
                    end
                end
            end
        end
    end

    return nothing
end 

function Iab(a::Patch, b::Patch, Δε::Float64, V_squared::Function)
    if abs(a.energy - b.energy) < Δε/2 
        return 16 * V_squared(a.momentum, b.momentum) * a.djinv * b.djinv / Δε 
    else
        return 0.0
    end
end

function electron_electron!(L::AbstractArray{<:Real,2}, grid::Vector{Patch}, Δε::Real, T::Real, hams::Vector{Function})
    f0s  = map(x -> f0(x.energy, T), grid) # Fermi-Dirac Grid

    ζ = MVector{6, Float64}(undef)
    Iibc = zeros(Float64, length(grid), length(grid))
    for i in ProgressBar(eachindex(grid))
        for j in eachindex(grid)
            @inbounds Iibc[j, :] .= sum.(Γabc!.(Ref(ζ), Ref(grid[i]), Ref(grid[j]), grid, Ref(T), Ref(Δε), Ref(hams)))
        end
        
        for j in eachindex(grid)
            i == j && continue
            @inbounds L[i,j] += dot(Iibc[j,:], (1 .- f0s)) * f0s[j] / (1 - f0(grid[i].energy, T)) 
            @inbounds L[i,j] -= 2 * dot(Iibc[:,j], f0s) * (1 - f0s[j]) / (1 - f0(grid[i].energy, T))
        end

        L[i, :] /= grid[i].dV
    end
end

function electron_electron!(L::AbstractArray{<:Real,2}, grid::Vector{Patch}, Δε::Real, T::Real, itps::Vector{ScaledInterpolation})
    f0s  = map(x -> f0(x.energy, T), grid)

    ζ = MVector{6, Float64}(undef)
    Iibc = zeros(Float64, length(grid), length(grid))
    for i in ProgressBar(eachindex(grid))
        for j in eachindex(grid)
            @inbounds Iibc[j, :] .= sum.(Γabc!.(Ref(ζ), Ref(grid[i]), Ref(grid[j]), grid, Ref(T), Ref(Δε), Ref(itps)))
        end
        
        for j in eachindex(grid)
            i == j && continue
            @inbounds L[i,j] += dot(Iibc[j,:], (1 .- f0s)) * f0s[j] / (1 - f0(grid[i].energy, T)) 
            @inbounds L[i,j] -= 2 * dot(Iibc[:,j], f0s) * (1 - f0s[j]) / (1 - f0(grid[i].energy, T))
        end

        L[i, :] /= grid[i].dV
    end
end

# function electron_electron!(L::AbstractArray{<:Real,2}, grid::Vector{Patch}, Δε::Real, T::Real, H::Function, N::Int)
#     itps = Ludwig.get_bands(H, N)
#     n_bands = length(itps) 

#     f0s  = SVector{length(grid), Float64}(map(x -> f0(x.energy, T), grid)) # Fermi-Dirac Grid
#     cfs  = 1 .- f0s # Complement distribution

#     # Preallocation of loop variables
#     ζ = MVector{6, Float64}(undef)
#     # integral = MVector{n_bands, Float64}(undef)
#     Ki = Matrix{SVector{n_bands, Float64}}(undef, length(grid), length(grid))
#     W = MMatrix{n_bands, n_bands, ComplexF64}(undef)
#     k1 = Matrix{SVector{2,Float64}}(undef, length(grid), length(grid))
#     F2mi_squared = MVector{n_bands, Float64}(undef)
#     Fmi_squared::Float64 = 0.0
#     Fji::ComplexF64 = 0.0
   
#     for i in eachindex(grid)
#         @show i
        
#         @time for j in eachindex(grid)
#             @inbounds Ki[j, :] .= f0s[j] * Γabc!.( Ref(ζ), Ref(grid[i]), Ref(grid[j]), grid, Ref(T), Ref(Δε), Ref(itps)) .* cfs
#             @inbounds k1[j, :] .= map_to_first_bz.([grid[i].momentum .+ grid[j].momentum] .- map(x -> x.momentum, grid))
#         end
        
#         @time for j in eachindex(grid)
#             i == j && continue

#             Fji = dot(grid[j].w, grid[i].w)
#             for m in eachindex(grid)

#                 ### ki + kj --> km + k4 ###
#                 Fmi_squared = abs2(dot(grid[m].w, grid[i].w))
#                 if Fmi_squared != 0
#                     L[i,j] += Fmi_squared * dot( abs2.(eigvecs(H(k1[j, m])) * conj(grid[j].w)) , Ki[j, m])
#                 end
            
#                 ### Depopulation Events ###
#                 @inbounds W .= conj.(eigvecs(H(k1[m,j])))

#                 F2mi_squared .=  abs2.(W * grid[i].w * dot(grid[j].w, grid[m].w)) .+ abs2.(W * grid[m].w * Fji)

#                 @inbounds L[i,j] -= dot(F2mi_squared, Ki[m, j])
#             end
#         end

#         L[i, :] /= (grid[i].dV * (1 - f0(grid[i].energy, T)) )
#     end
# end

# function electron_electron!(L::AbstractArray{<:Real,2}, grid::Vector{Patch}, Δε::Real, T::Real, H::Function, N::Int)
#     itps = Ludwig.get_bands(H, N)
#     n_bands = length(itps) 

#     f0s  = SVector{length(grid), Float64}(map(x -> f0(x.energy, T), grid)) # Fermi-Dirac Grid
#     F2mi_squared = MVector{n_bands, Float64}(undef)
    
#     Threads.@threads for i in eachindex(grid)
#         @show i
        
#         K = Array{Float64}(undef, length(grid), length(grid), n_bands)
#         W = MMatrix{n_bands, n_bands, ComplexF64}(undef)

#         @time ee_kernel!(K, grid, i, T, Δε, itps, f0s)

#         Fjm_squared::Float64 = 0.0
#         Fni_squared = map(x -> abs2(dot(x.w, grid[i].w)), grid)

#         @time for j in eachindex(grid)
#             i== j && continue
#             L[i,j] = 0.0
            
#             for m in eachindex(grid)
#                 if Fni_squared[m] != 0
#                     L[i,j] += Fni_squared[m] * dot(abs2.(eigvecs(H(grid[i].momentum + grid[j].momentum - grid[m].momentum)) * grid[j].w), @view K[j,m,:])
#                 end

#                 W = eigvecs(H(grid[i].momentum + grid[m].momentum - grid[j].momentum))
#                 @show W
#                 return nothing

#                 Fjm_squared = abs2(dot(grid[j].w, grid[m].w))
#                 for μ in 1:n_bands
#                     F2mi_squared[μ] = abs2(dot(W[:, μ], grid[i].w)) * Fjm_squared + abs2(dot(W[:, μ], grid[m].w)) * Fni_squared[j]
#                 end

#                 L[i,j] -= dot(F2mi_squared, @view K[m, j, :])

#             end
#         end

#         L[i, :] /= (grid[i].dV * (1 - f0(grid[i].energy, T)) )
#     end
# end


# cf_eignvecs is a function which populates a matrix with closed form solutions for the eigenvectors
# function electron_electron!(L::AbstractArray{<:Real,2}, grid::Vector{Patch}, Δε::Real, T::Real, H::Function, N::Int, cf_eigvecs!::Function)
#     itps = Ludwig.get_bands(H, N)
#     n_bands = length(itps) 
#     f0s  = SVector{length(grid), Float64}(map(x -> f0(x.energy, T), grid)) # Fermi-Dirac Grid
    
#     F_squared = Matrix{Float64}(undef, length(grid), length(grid))
#     for j in eachindex(grid)
#         for m in eachindex(grid)
#             F_squared[j,m] = dot(grid[j].w, grid[m].w)^2
#         end 
#     end

#     Threads.@threads for i in eachindex(grid)
#         K = Array{Float64}(undef, length(grid), length(grid), n_bands)
#         W = MMatrix{n_bands, n_bands, Float64}(undef)
#         ee_kernel!(K, grid, i, T, Δε, itps, f0s)

#         temp1::Float64 = 0.0
#         temp2::Float64 = 0.0

#         vertex(w, j, m) = begin
#             temp1 = 0.0
#             for μ in 1:n_bands
#                 temp2 = 0.0
#                 for σ in 1:n_bands
#                     temp2 += W[σ, μ] * w[σ]
#                 end
#                 @inbounds temp1 += temp2^2 * K[j,m,μ]
#             end
#             nothing
#         end

#         wi = grid[i].w
#         wj = MVector{n_bands, Float64}(undef); fill!(wj, 0.0)
#         wm = MVector{n_bands, Float64}(undef); fill!(wm, 0.0)

#         for j in eachindex(grid)
#             i == j && continue
#             L[i,j] = 0.0

#             for μ in 1:n_bands
#                 wj[μ] = grid[j].w[μ] # Reduced memory allocation
#             end 
            
#             kij = grid[i].momentum + grid[j].momentum
#             qij = grid[i].momentum - grid[j].momentum

#             for m in eachindex(grid)

#                 if F_squared[m, i] != 0
#                     cf_eigvecs!(W, kij - grid[m].momentum)
#                     vertex(wj, j, m)
#                     L[i,j] += F_squared[m, i] * temp1
#                 end

#                 if F_squared[j, i] != 0 || F_squared[j, m] != 0

#                     cf_eigvecs!(W, qij + grid[m].momentum)

#                     if F_squared[j, m] != 0
#                         vertex(wi, m, j)
#                         L[i,j] -= F_squared[j, m] * temp1
#                     end

#                     if F_squared[j, i] != 0
#                         for μ in 1:n_bands
#                             wm[μ] = grid[m].w[μ]
#                         end
                        
#                         vertex(wm, m, j)
#                         L[i,j] -= F_squared[j, i] * temp1
#                     end
#                 end
#             end
#         end

#         L[i, :] /= (grid[i].dV * (1 - f0s[i]) )
#     end
# end

# function electron_electron(grid::Vector{Patch}, i::Int, j::Int, Δε::Real, T::Real, H::Function, N::Int, cf_eigvecs!::Function)
#     Lij::Float64 = 0.0
#     f0s = map(x -> f0(x.energy, T), grid) # Fermi-Dirac Grid

#     for m in eachindex(grid)
        
#     end

#     return Lij / grid[i].dV
# end

function electron_impurity!(L::AbstractArray{<:Real,2}, grid::Vector{Patch}, Δε::Real, V_squared)
    for i in ProgressBar(eachindex(grid))        
        for j in eachindex(grid)
            i == j && continue
            @inbounds L[i,j] -= Iab(grid[i], grid[j], Δε, V_squared)
        end

        L[i, :] /= grid[i].dV
    end
    return nothing
end

function electron_impurity!(L::AbstractArray{<:Real,2}, grid::Vector{Patch}, Δε::Real, V_squared::Matrix)
    ℓ = size(Γ)[1] ÷ 3
    for i in ProgressBar(eachindex(grid))        
        for j in eachindex(grid)
            i == j && continue
            @inbounds L[i,j] -= Iab(grid[i], grid[j], Δε, (x,y) -> V_squared[(i-1)÷ℓ + 1, (j-1)÷ℓ + 1])
        end

        L[i, :] /= grid[i].dV
    end
    return nothing
end