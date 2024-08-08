
const ρ::Float64 = 4*6^(1/3)/pi

"Volume of the 5-dimensional unit sphere"
const vol::Float64 = (8 * pi^2 / 15)

"""
    Γabc!(ζ, a, b, c, T, Δε, hams::Vector{Function})

Compute the integral
```math
    \\mathcal{K}^\\mu_{abc} \\equiv \\sum_{\\mu} \\int_a \\int_b \\int_c (1 - f^{(0)}(\\mathbf{k}_a + \\mathbf{k}_b + \\mathbf{k}_c)) \\delta(\\varepsilon_a + \\varepsilon_b - \\varepsilon_c - \\varepsilon^\\mu(\\mathbf{k}_a + \\mathbf{k}_b - \\mathbf{k}_c)) 
```
where ``\\mu`` denotes the band index of the dispersion in `hams` and 
```math
    \\int_i \\equiv \\frac{1}{a^2} \\int_{\\mathbf{k} \\in \\mathcal{P}_i} d^2\\mathbf{k}
```
is an integral over momenta in patch ``\\mathcal{P}_i``. 

"""
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

"""
    Γabc!(ζ, a, b, c, T, Δε, h::Function)

When only a single band `h` is provided, compute the integration kernel without summing over bands. 
"""
function Γabc!(ζ::MVector{6, Float64}, a::Patch, b::Patch, c::Patch, T::Real, d, εabc, ε::Function)

    δ = energy(a) + energy(b) - energy(c) - εabc # Energy conservation violations

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
    u::SVector{6, Float64} = [a.de/2.0, 0.0, b.de/2.0, 0.0, -c.de/2.0, 0.0] - ζ

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

function Γabc!(ζ::MVector{6, Float64}, a::Patch, b::Patch, c::Patch, T::Real, d, εabc, itp::ScaledInterpolation)
    δ = energy(a) + energy(b) - energy(c) - εabc # Energy conservation violations

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
    u::SVector{6, Float64} = [a.de/2.0, 0.0, b.de/2.0, 0.0, -c.de/2.0, 0.0] - ζ

    if abs(δ) < a.de * 1e-4
        return vol * a.djinv * b.djinv * c.djinv * ρ^(5/2) * (1 - f0(εabc, T)) / norm(u)
    else
        # εabc ≈ ε0 + ζ . x
        ρ < δ^2 / dot(u,u) && return 0.0 # Check for intersection of energy conserving 5-plane with coordinate space

        xpara::SVector{6, Float64} = - δ * u / dot(u,u) # Linearized coordinate along energy conserving direction

        r5::Float64 = (ρ - δ^2 / dot(u,u) )^(5/2)

        return vol * a.djinv * b.djinv * c.djinv * r5 * (1 - f0(εabc + dot(ζ, xpara), T)) / norm(u)
    end
    

end

"""
    Γabc!(ζ, a, b, c, T, Δε, itps::Vector{ScaledInterpolation})

The same integral where energies are evaluated by interpolation of the bands over the 1st Brillouin Zone
"""
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

"""
    ee_kernel!(K, grid, i, T, Δε, itps, f0s)

Populate the preallocated array `K` with the values of ``\\mathcal{K}^\\mu_{ijm}`` for the row `i` of the collision operator.

Energies are calculated from `itps`, the interpolation of each band and `f0s` is a vector of the Fermi-Dirac distribution evaluated for the energy of each patch in `grid` at temperature `T`.
"""
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

"""
    electron_electron!(L, grid, Δε, T, hams::Vector{Function})

Populate the collision matrix `L` defined on patches in `grid` assuming equal weight for interband scattering given explicit functions for each band in `hams`.
"""
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

"""
    electron_electron!(L, grid, Δε, T, itps::Vector{ScaledInterpolation})

Populate the collision matrix `L` defined on patches in `grid` assuming equal weight for interband scattering given gridded interpolations of bands in `itps`.
"""
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

"""
    electron_electron!(L, grid, Δε, T, H, N, cf_eigvecs!)

Populate the collision matrix `L` defined on patches in `grid` for a scattering perturbation to the Hamiltonian of the form
```math
V = \\frac{U}{N} \\sum_{a,b} \\sum_{\\mathbf{k}_1, \\mathbf{k}_2, \\mathbf{q}} \\sum_{\\mu\\nu\\sigma\\tau} c^\\dagger_{\\mathbf{k}_1 - \\mathbf{q},\\sigma} \\left( W_{\\mathbf{k}_1 - \\mathbf{q}}^{a \\sigma}\\right)^* W_{\\mathbf{k}_1}^{a \\mu} c_{\\mathbf{k}_1, \\mu} c^\\dagger_{\\mathbf{k}_2 + \\mathbf{q},\\tau} \\left( W_{\\mathbf{k}_2 + \\mathbf{q}}^{b \\tau}\\right)^* W_{\\mathbf{k}_2}^{b \\nu} c_{\\mathbf{k}_2, \\nu}
```
The field `w` of the type `Patch` gives the column of ``W_\\mathbf{k}`` corresponding to the field `band_index` evaluated at the field `band_index`.

The function `cf_eigvecs!` is a convenience function that speeds up performance by populating a pre-allocated matrix with the closed form solution for the eigenvectors of `H`.
"""
function electron_electron!(L::AbstractArray{<:Real,2}, grid::Vector{Patch}, Δε::Real, T::Real, H::Function, N::Int, cf_eigvecs!::Function)
    itps = Ludwig.get_bands(H, N)
    n_bands = length(itps) 
    f0s  = SVector{length(grid), Float64}(map(x -> f0(x.energy, T), grid)) # Fermi-Dirac Grid
    
    F_squared = Matrix{Float64}(undef, length(grid), length(grid))
    for j in eachindex(grid)
        for m in eachindex(grid)
            F_squared[j,m] = dot(grid[j].w, grid[m].w)^2
        end 
    end

    Threads.@threads for i in eachindex(grid)
        K = Array{Float64}(undef, length(grid), length(grid), n_bands)
        W = MMatrix{n_bands, n_bands, Float64}(undef)
        ee_kernel!(K, grid, i, T, Δε, itps, f0s)

        temp1::Float64 = 0.0
        temp2::Float64 = 0.0

        vertex(w, j, m) = begin
            temp1 = 0.0
            for μ in 1:n_bands
                temp2 = 0.0
                for σ in 1:n_bands
                    temp2 += W[σ, μ] * w[σ]
                end
                @inbounds temp1 += temp2^2 * K[j,m,μ]
            end
            nothing
        end

        wi = grid[i].w
        wj = MVector{n_bands, Float64}(undef); fill!(wj, 0.0)
        wm = MVector{n_bands, Float64}(undef); fill!(wm, 0.0)

        for j in eachindex(grid)
            i == j && continue
            L[i,j] = 0.0

            for μ in 1:n_bands
                wj[μ] = grid[j].w[μ] # Reduced memory allocation
            end 
            
            kij = grid[i].momentum + grid[j].momentum
            qij = grid[i].momentum - grid[j].momentum

            for m in eachindex(grid)

                if F_squared[m, i] != 0
                    cf_eigvecs!(W, kij - grid[m].momentum)
                    vertex(wj, j, m)
                    L[i,j] += F_squared[m, i] * temp1
                end

                if F_squared[j, i] != 0 || F_squared[j, m] != 0

                    cf_eigvecs!(W, qij + grid[m].momentum)

                    if F_squared[j, m] != 0
                        vertex(wi, m, j)
                        L[i,j] -= F_squared[j, m] * temp1
                    end

                    if F_squared[j, i] != 0
                        for μ in 1:n_bands
                            wm[μ] = grid[m].w[μ]
                        end
                        
                        vertex(wm, m, j)
                        L[i,j] -= F_squared[j, i] * temp1
                    end
                end
            end
        end

        L[i, :] /= (grid[i].dV * (1 - f0s[i]) )
    end
end

function get_ebounds(p::Patch)
    return [p.energy - p.de/2, p.energy + p.de/2]
end

function electron_electron(grid::Vector{Patch}, f0s::Vector{Float64}, i::Int, j::Int, itps::Vector{ScaledInterpolation}, T::Real, Fpp::Function, Fpk::Function)
    Lij::Float64 = 0.0
    w123::Float64 = 0.0
    w124::Float64 = 0.0

    ζ  = MVector{6,Float64}(undef)

    kij = grid[i].momentum + grid[j].momentum
    qij = grid[i].momentum - grid[j].momentum
    kijm = Vector{Float64}(undef, 2)
    qimj = Vector{Float64}(undef, 2)

    energies = Vector{Float64}(undef, length(itps))

    iplusj_bounds = grid[i].bounds .+ grid[j].bounds
    iminusj_bounds = grid[i].bounds .- reverse(grid[j].bounds)
    m_bounds = Vector{Float64}(undef, 2)
    mn_bounds = Vector{Float64}(undef, 2)
    ijm_bounds = Vector{Float64}(undef, 2)
    imj_bounds = Vector{Float64}(undef, 2)

    μ4::Int = 0
    μ34::Int = 0
    min_e::Float64 = 0

    for m in eachindex(grid)
        m_bounds .= grid[m].bounds
        kijm .= mod.(kij .- grid[m].momentum .+ 0.5, 1.0) .- 0.5
        
        min_e = 1e3
        μ4 = 1
        for μ in eachindex(itps)
            energies[μ] = itps[μ](kijm[1], kijm[2])
            if abs(energies[μ]) < min_e 
                min_e = abs(energies[μ]); μ4 = μ
            end
        end

        mn_bounds .= m_bounds .+ [energies[μ4] - grid[m].de, energies[μ4] + grid[m].de]

        ijm_bounds[1] = iplusj_bounds[1] - mn_bounds[2]
        ijm_bounds[2] = iplusj_bounds[2] - mn_bounds[1]

        if sign(ijm_bounds[1]) != sign(ijm_bounds[2]) 
            w123 = Weff_squared_123(grid[i], grid[j], grid[m], Fpp, Fpk, kijm, μ4)

            if w123 != 0
                Lij += w123 * Γabc!(ζ, grid[i], grid[j], grid[m], T, kijm, energies[μ4], itps[μ4]) * f0s[j] * (1 - f0s[m])
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

        mn_bounds .= m_bounds .- [energies[μ34] + grid[m].de, energies[μ34] - grid[m].de]
        imj_bounds .= iminusj_bounds .+ mn_bounds
        if sign(imj_bounds[1]) != sign(imj_bounds[2]) 
            w123 = Weff_squared_123(grid[i], grid[m], grid[j], Fpp, Fpk, qimj, μ34)
            w124 = Weff_squared_124(grid[i], grid[m], grid[j], Fpp, Fpk, qimj, μ34)

            if w123 + w124 != 0
                Lij -= (w123 + w124) * Γabc!(ζ, grid[i], grid[m], grid[j], T, qimj, energies[μ34], itps[μ34]) * f0s[m] * (1 - f0s[j])
            end
        end

    end

    return Lij * f0s[i]
end

# function Iab(a::Patch, b::Patch, Δε::Float64, V_squared::Function)
#     if abs(a.energy - b.energy) < Δε/2 
#         return 16 * V_squared(a.momentum, b.momentum) * a.djinv * b.djinv / Δε 
#     else
#         return 0.0
#     end
# end

# function electron_impurity!(L::AbstractArray{<:Real,2}, grid::Vector{Patch}, Δε::Real, V_squared)
#     for i in ProgressBar(eachindex(grid))        
#         for j in eachindex(grid)
#             i == j && continue
#             @inbounds L[i,j] -= Iab(grid[i], grid[j], Δε, V_squared)
#         end

#         L[i, :] /= grid[i].dV
#     end
#     return nothing
# end

# function electron_impurity!(L::AbstractArray{<:Real,2}, grid::Vector{Patch}, Δε::Real, V_squared::Matrix)
#     ℓ = size(Γ)[1] ÷ 3
#     for i in ProgressBar(eachindex(grid))        
#         for j in eachindex(grid)
#             i == j && continue
#             @inbounds L[i,j] -= Iab(grid[i], grid[j], Δε, (x,y) -> V_squared[(i-1)÷ℓ + 1, (j-1)÷ℓ + 1])
#         end

#         L[i, :] /= grid[i].dV
#     end
#     return nothing
# end