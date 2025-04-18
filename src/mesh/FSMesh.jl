module FSMesh

import StaticArrays: SVector, SMatrix
using LinearAlgebra
import ..MarchingSquares: Isoline, get_bounding_box, contours, contour_intersection
import ..Lattices: Lattice, get_ibz, point_group
import ..Groups: get_matrix_representation
using ..GeometryUtilities
import ForwardDiff: gradient, derivative

export Patch, VirtualPatch, AbstractPatch, energy, momentum, velocity
export Mesh, patches, corners, corner_indices
export mesh_region, ibz_mesh, bz_mesh, circular_fs_mesh

abstract type AbstractPatch end

"""
    Patch(e::Float64, k::SVector{2,Float64}, v::SVector{2,Float64}, de::Float64, dV::Float64, jinv::Matrix{Float64}, djinv::Float64, band_index::Int)
Construct a `Patch' object defining regions of momentum space over which to integrate. 
# Fields
- `e`: energy
- `k`: momentum 
- `v`: group velocity
- `de`: width of patch in energy
- `dV`: area of patch in momentum space
- `jinv`: Jacobian of transformation from (kx, ky) --> (E, s)
- `djinv`: determinant of above Jacobian
- `band_index`: index of band from which `e` was sampled at `k`
"""
struct Patch <: AbstractPatch
    e::Float64 # Energy
    k::SVector{2,Float64} # Momentum
    v::SVector{2,Float64} # Group velocity
    de::Float64
    dV::Float64 # Patch area
    jinv::Matrix{Float64} # Jacobian of transformation from (kx, ky) --> (E, S)
    djinv::Float64
    band_index::Int
end

"""
    VirtualPatch(e::Float64, k::SVector{2,Float64}, v::SVector{2,Float64}, band_index::Int)
Construct a `VirtualPatch' object that can be operated on as if it were a `Patch` for the purposes of sampling momentum, energy, and group velocity but which cannot be integrated over. 
# Fields
- `e`: energy
- `k`: momentum 
- `v`: group velocity
- `band_index`: index of band from which `e` was sampled at `k`
"""
struct VirtualPatch <: AbstractPatch
    e::Float64 # Energy
    k::SVector{2,Float64} # Momentum
    v::SVector{2,Float64} # Group velocity
    band_index::Int
end

energy(p::AbstractPatch)   = p.e
momentum(p::AbstractPatch) = p.k
velocity(p::AbstractPatch) = p.v
band(p::AbstractPatch)     = p.band_index

area(p::Patch)             = p.dV

function patch_op(p::Patch, M::Matrix)
    return Patch(
        p.e,
        SVector{2}(M * p.k), 
        SVector{2}(M * p.v),
        p.de,
        p.dV,
        M * p.jinv, 
        p.djinv,
        p.band_index
    )
end

function patch_op(p::VirtualPatch, M::Matrix)
    return VirtualPatch(
        p.energy,
        SVector{2}(M * p.momentum), 
        SVector{2}(M * p.v),
        band_index
    )
end

"""
    Mesh(patches::Vector{Patch}, corners::Vector{SVector{2,Float64}}, corner_inds::Vector{SVector{4,Int}}})
Construct a container struct of patches which contains information for plotting functions defined on patch centers.
# Fields
- `patches`: Vector of patches
- `corners`: Vector of points on patch corners for plotting mesh
- `corner_inds`: Vector of vector of indices in the `corners` vector corresponding to the corners of each patch in `patches`
"""
struct Mesh
    patches::Vector{Patch}
    corners::Vector{SVector{2, Float64}} 
    corner_inds::Vector{SVector{4, Int}}
end

patches(m::Mesh) = m.patches
corners(m::Mesh) = m.corners
corner_indices(m::Mesh) = m.corner_inds

###
# Nonuniform Energy Gridding
###

function populate_abscissas!(x, δ)
    Δx = 0.0
    for i ∈ eachindex(x)
        if i == 1
            x[i] = δ
        else
            x[i] = x[i-1] + Δx
        end
        Δx = 4 * δ * cosh(x[i]/2)^2
    end
    return nothing
end

function get_abscissas(n, α, threshold = 1e-10, max_iter = 1000)
    x = Vector{Float64}(undef, n)

    ϵ = α / n
    iter = 1
    step = 0.1 * ϵ^(1/3)
    populate_abscissas!(x, ϵ^(1/3))
    old_sign = sign(x[end] - α)

    while abs(x[end] - α) >= threshold && iter <= max_iter
        new_sign = sign(x[end] - α)
        if new_sign != old_sign
            old_sign = new_sign
            step /= 10.0
        end
        while ϵ - new_sign * step < 0
            step /= 2.0
        end
        ϵ += - new_sign * step
        populate_abscissas!(x, ϵ^(1/3))

        iter += 1
    end

    return vcat(-reverse(x), x)#, ϵ
end

###
# General IBZ Meshes
###

function get_arclengths(curve)
    s = 0.0
    arclengths = Vector{Float64}(undef, length(curve))
    arclengths[1] = 0.0
    for i in eachindex(curve)
        i == 1 && continue
        s += norm(curve[i] - curve[i-1])
        arclengths[i] = s
    end
    return arclengths
end

function arclength_slice(iso::Isoline, n::Int)
    curve = iso.points

    arclengths = get_arclengths(curve)
    
    τ = LinRange(0.0, arclengths[end], n)

    grid = [curve[begin]]

    j = 1
    jmax = length(arclengths)
    for i in eachindex(τ)
        i == 1 && continue
        while j < jmax && arclengths[j] <= τ[i]
            j += 1
        end

        push!(grid, curve[j-1] + (curve[j] - curve[j - 1]) * (τ[i] - arclengths[j-1]) / (arclengths[j] - arclengths[j-1]))
    end

    return grid, collect(τ)   
end

function mesh_region(region, ε, band_index::Int, T, n_levels::Int, n_cuts::Int, N = 1001, α = 6.0)
    n_levels = max(3, n_levels) # Enforce minimum of 2 patches in the energy direction
    n_cuts = max(3, n_cuts) # Enforce minimum of 2 patches in angular direction 

    ((x_min, x_max), (y_min, y_max)) = get_bounding_box(region)

    X = LinRange(x_min, x_max, N)
    Δx = (x_max - x_min) / (N - 1)
    Ny = round(Int, (y_max - y_min) / Δx)
    Y = LinRange(y_min, y_max, Ny)
    E = Matrix{Float64}(undef, N, Ny) 
    for (i,x) in enumerate(X)
        for (j,y) in enumerate(Y)
            k = [x,y]
            E[i,j] = ε(k)
        end
    end

    e_threshold = α*T # Half-width of Fermi tube
    e_min = max(-e_threshold, 0.999 * minimum(E[.!isnan.(E)]))
    e_max = min(e_threshold, 0.999 * maximum(E[.!isnan.(E)]))
    energies = collect(LinRange(e_min, e_max, n_levels))
    # Shift energy levels to include corner energy if there is a crossing
    for corner_e ∈ ε.(region)
        if e_min < corner_e < e_max 
            i = argmin(abs.(energies .- corner_e))
            energies[1:i] = LinRange(e_min, corner_e, i)
            Δe = (e_max - corner_e) / (n_levels - i)
            energies[i+1:end] = LinRange(corner_e + Δe, e_max, n_levels - i)
        end 
    end

    foliated_energies = Vector{Float64}(undef, 2 * n_levels - 1)

    for i in eachindex(foliated_energies)
        if isodd(i)
            foliated_energies[i] = energies[(i - 1) ÷ 2 + 1]
        else
            foliated_energies[i] = (energies[i ÷ 2 + 1] + energies[i ÷ 2]) / 2
        end
    end

    c = contours(X, Y, E, foliated_energies; mask = region) # Generate Fermi surface contours

    patches = Matrix{Patch}(undef, n_levels-1, n_cuts-1)
    corners = Vector{SVector{2, Float64}}(undef, (2 * n_levels - 2) * n_cuts)
    corner_ids = Matrix{SVector{4, Int}}(undef, n_levels-1, n_cuts-1)

    cind = 1 # Corner iteration index

    for i in 2:2:2*n_levels-1
        k, arclengths = arclength_slice(c[i].isolines[1], 2 * n_cuts - 1)

        # corners[cind], ij3 = contour_intersection(k[1], gradient(ε, k[1]), c[i-1].isolines[1])
        # corners[cind+1], ij4 = contour_intersection(k[1], gradient(ε, k[1]), c[i+1].isolines[1])
        endpoints = [c[i-1].isolines[1].points[begin], c[i-1].isolines[1].points[end]]
        i3 = argmin(map(x -> norm(x .- k[1]), endpoints))
        ij3 = (i3, i3)
        corners[cind] = endpoints[i3]

        endpoints = [c[i+1].isolines[1].points[begin], c[i+1].isolines[1].points[end]]
        i4 = argmin(map(x -> norm(x .- k[1]), endpoints))
        ij4 = (i4, i4)
        corners[cind+1] = endpoints[i4]


        cind += 2        
        for j in 2:2:lastindex(k)
            corners[cind], ij1 = contour_intersection(k[j+1], gradient(ε, k[j+1]), c[i-1].isolines[1])
            corners[cind+1], ij2 = contour_intersection(k[j+1], gradient(ε, k[j+1]), c[i+1].isolines[1])
            corner_ids[i÷2,j÷2] = [cind-2, cind-1, cind+1, cind]
            cind += 2

            i3, i1 = map(x -> x[argmin(
                [norm(k[j] - c[i-1].isolines[1].points[x[1]]),
                norm(k[j] - c[i-1].isolines[1].points[x[2]])])],
                (ij3, ij1)
            )
            i4, i2 = map(x -> x[argmin(
                [norm(k[j] - c[i+1].isolines[1].points[x[1]]),
                norm(k[j] - c[i+1].isolines[1].points[x[2]])])],
                (ij4, ij2)
            )
            

            dV = poly_area(vcat(corners[cind-4:cind-1], c[i-1].isolines[1].points[min(i1,i3):max(i1,i3)], c[i+1].isolines[1].points[min(i2,i4):max(i2,i4)]), k[j])

            # dV = dV2

            ij3 = ij1; ij4 = ij2

            Δε = foliated_energies[i+1] - foliated_energies[i-1]
            Δs = arclengths[j+1] - arclengths[j-1]

            v = gradient(ε, k[j])

            p1, _ = contour_intersection(k[j], gradient(ε, k[j]), c[i-1].isolines[1])
            p2, _ = contour_intersection(k[j], gradient(ε, k[j]), c[i+1].isolines[1])

            A = Matrix{Float64}(undef, 2, 2) 
            J = Matrix{Float64}(undef, 2, 2)

            A[1,1] = (p2[1] - p1[1]) / Δε # ∂kx/∂ε |s
            A[1,2] = (p2[2] - p1[2]) / Δε # ∂ky/∂ε |s
            A[2,1] = (k[j+1][1] - k[j-1][1]) / Δs   # ∂kx/∂s |ε
            A[2,2] = (k[j+1][2] - k[j-1][2]) / Δs   # ∂ky/∂s |ε
            
            J[1,1] = 2 * v[1] / Δε # ∂ε/∂kx |ky
            J[1,2] = 2 * v[2] / Δε # ∂ε/∂ky |kx
            J[2,1] = - 2 * A[1,2] / det(A) / Δs   # ∂s/∂kx |ky
            J[2,2] = 2 * A[1,1] / det(A) / Δs # ∂s/∂ky |kx

            patches[i÷2,j÷2] = Patch(
                foliated_energies[i],
                k[j], 
                v,
                Δε, 
                dV,
                inv(J),
                1/abs(det(J)),
                band_index
            )
        end

    end

    return Mesh(vec(patches), corners, vec(corner_ids))
end

mesh_region(region, ε, T, n_levels, n_cuts, N = 1001, α = 6.0) = mesh_region(region, ε, 1, T, n_levels, n_cuts, N, α)

function ibz_mesh(l::Lattice, bands::AbstractVector, T, n_levels::Int, n_cuts::Int, N::Int = 1001, α::Real = 6.0)
    full_patches = Vector{Patch}(undef, 0)
    full_corners = Vector{SVector{2, Float64}}(undef, 0)
    full_corner_inds = Vector{SVector{4, Int}}
    ibz = get_ibz(l)

    for i in eachindex(bands)
        mesh = mesh_region(ibz, bands[i], i, T, n_levels, n_cuts, N, α)
        full_patches = vcat(full_patches, mesh.patches)
        ℓ = length(full_corners)
        full_corner_inds = vcat(full_corners, map(x -> SVector{4, Int}(x .+ ℓ), mesh.corner_inds))
        full_corners = vcat(full_corners, mesh.corners)
    end

    return Mesh(full_patches, full_corners, full_corner_inds)
end

ibz_mesh(l::Lattice, ε::Function, T, n_levels::Int, n_cuts::Int, N::Int = 1001, α::Real = 6.0) = ibz_mesh(l, [ε], T, n_levels, n_cuts, N, α)

function bz_mesh(l::Lattice, bands::AbstractVector, T, n_levels::Int, n_cuts::Int, N::Int = 1001, α::Real = 6.0)
    G = point_group(l)
    ibz = get_ibz(l)

    θs = Vector{Float64}(undef, length(G.elements))
    for i in eachindex(G.elements)
        O = get_matrix_representation(G.elements[i])
        k = sum(map(x -> O*x, ibz)) # Central direction vector of IBZ
        θs[i] = mod(atan(k[2], k[1]), 2pi)
    end
    θperm = sortperm(θs)

    full_patches = Matrix{Patch}(undef, n_levels - 1, 0)
    full_corners = Vector{SVector{2, Float64}}(undef, 0)
    full_corner_inds = Matrix{SVector{4, Int}}(undef, n_levels - 1, 0)

    for j in eachindex(bands)
        mesh = mesh_region(ibz, bands[j], j, T, n_levels, n_cuts, N, α)
        
        for i in θperm
            O = get_matrix_representation(G.elements[i])
            ℓ = length(full_corners)
            if det(O) < 0.0 # Improper rotation
                full_patches = hcat(full_patches, 
                    reverse( map(x -> patch_op(x, O), reshape(mesh.patches, n_levels - 1, :)), dims = 2)
                )
                full_corner_inds = hcat(full_corner_inds, reverse( map(x -> SVector{4, Int}(x .+ ℓ), reshape(mesh.corner_inds,  n_levels - 1, :)), dims = 2))
            else
                full_patches = hcat(full_patches, 
                    map(x -> patch_op(x, O), reshape(mesh.patches, n_levels - 1, :)) 
                )
                full_corner_inds = hcat(full_corner_inds, map(x -> SVector{4, Int}(x .+ ℓ), reshape(mesh.corner_inds,  n_levels - 1, :)))
            end
            
            
            full_corners = vcat(full_corners, map(x -> O*x, mesh.corners))
        end
    end

    return Mesh(vec(full_patches), vec(full_corners), vec(full_corner_inds))
end

bz_mesh(l::Lattice, ε, T, n_levels::Int, n_cuts::Int, N::Int = 1001, α::Real = 6.0) = bz_mesh(l, [ε], T, n_levels, n_cuts, N, α)

function secant_method(f, x0, x1, maxiter; atol = eps(Float64))
    x2 = 0.0
    for i in 1:maxiter
        x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        x0, x1 = x1, x2
        abs(f(x2)) < atol && break
    end

    return x2
end

function circular_fs_mesh(ε, T::Real, n_levels::Int, n_angles::Int, α::Real = 6.0; maxiter = 10000, atol = 1e-10, nonuniform = false)
    n_levels = max(4, n_levels) # Enforce minimum of 3 patches in the energy direction
    if isodd(n_levels)
        n_levels += 1
    end
    n_angles = max(3, n_angles) # Enforce minimum of 2 patches in angular direction
    Δθ = 2π / (n_angles - 1) 
    Δε = 2*α*T / (n_levels - 1) # From uniform spacing for sense of energy scale
    
    # Slice contours to generate patch corners
    corners = Matrix{SVector{2,Float64}}(undef, n_levels, n_angles)
    k = Matrix{SVector{2,Float64}}(undef, n_levels-1, n_angles-1)
    corner_indices = Matrix{SVector{4,Int}}(undef, n_levels-1, n_angles-1)
    radius = Vector{Float64}(undef, 2*n_levels - 1)
    if nonuniform
        energies = get_abscissas(n_levels ÷ 2, α) * T
    else
        energies = LinRange(-α, α, n_levels) * T
    end

    for i ∈ 1:2*n_levels-1
        if isodd(i)
            E = energies[(i+1)÷2]
        else
            E = (energies[i÷2+1] + energies[i÷2]) / 2.0
        end

        r0 = i == 1 ? 0.0 : radius[i-1]
        r1 = r0 + Δε / derivative(ε, r0)
        isinf(r1) && (r1 = Δε)

        radius[i] = secant_method(r -> ε(r) - E, r0, r1, maxiter; atol)
        if isodd(i)
            corners[i÷2 + 1, :] = map(θ -> SVector{2, Float64}(radius[i]*cos(θ), radius[i]*sin(θ)), LinRange(0, 2π, n_angles))
        else
            k[i÷2, :] = map(θ -> SVector{2, Float64}(radius[i]*cos(θ), radius[i]*sin(θ)), (Δθ/2):Δθ:2π)
            for j in 1:n_angles-1
                ref_index = (j - 1) * n_levels + (i÷2)
                corner_indices[i÷2, j] = [
                    ref_index, 
                    ref_index + 1,
                    ref_index + n_levels + 1,
                    ref_index + n_levels,
                    ]
            end
        end
    end

    v = map(x -> derivative(ε, norm(x)) * x / norm(x), k)

    J = zeros(Float64, 2, 2)
    patches = Matrix{Patch}(undef, n_levels-1, n_angles-1)
    for i in 1:size(patches)[1]
        Δε = energies[i+1] - energies[i] 
        dV = abs(0.5 * Δθ * (radius[2*i + 1]^2 - radius[2*i - 1]^2))
        for j in 1:size(patches)[2]
            θ = atan(k[i,j][2], k[i,j][1])

            J[1,1] = cos(θ) / norm(v[i,j])
            J[1,2] = sin(θ) / norm(v[i,j])
            J[2,1] = - norm(k[i,j]) * sin(θ)
            J[2,2] = norm(k[i,j]) * cos(θ)

            J = inv(J)
            J[1, :] *= 2/Δε
            J[2, :] *= 2/Δθ

            patches[i, j] = Patch(
                ε(norm(k[i,j])),   
                k[i,j], 
                v[i,j],
                Δε,
                dV,
                inv(J),
                1 / abs(det(J)),
                1
            )
        end
    end

    return Mesh(vec(patches), vec(corners), vec(corner_indices))
end

end