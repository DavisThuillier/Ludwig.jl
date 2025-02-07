module FSMesh

import StaticArrays: SVector, SMatrix
using LinearAlgebra
import ..MarchingSquares: Isoline, get_bounding_box, contours, contour_intersection
import ..Lattices: Lattice, get_ibz, in_polygon, point_group
import ForwardDiff: gradient

export MultibandPatch, SinglebandPatch, Patch

abstract type AbstractPatch end

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

struct SimplePatch <: AbstractPatch
    e::Float64 # Energy
    k::SVector{2,Float64} # Momentum
    v::SVector{2,Float64} # Group velocity
    band_index::Int
end

function patch_op(p::MultibandPatch, M::Matrix)
    return MultibandPatch(
        p.energy,
        SVector{2}(M * p.momentum), 
        SVector{2}(M * p.v),
        p.de,
        p.dV,
        M * p.jinv, 
        p.djinv,
        p.U,
        p.band_index
    )
end

function patch_op(p::SinglebandPatch, M::Matrix)
    return SinglebandPatch(
        p.energy,
        SVector{2}(M * p.momentum), 
        SVector{2}(M * p.v),
        p.de,
        p.dV,
        M * p.jinv, 
        p.djinv,
    )
end

"""
Container struct for patches over which to integrate.

# Fields
- `patches`: Vector of patches
- `corners`: Vector of points on patch corners for plotting mesh
"""
struct Mesh
    patches::Vector{Patch}
    corners::Vector{SVector{2, Float64}} 
    corner_inds::Vector{Vector{Int}}
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
        while j < jmax && arclengths[j] < τ[i]
            j += 1
        end

        push!(grid, curve[j-1] + (curve[j] - curve[j - 1]) * (τ[i] - arclengths[j-1]) / (arclengths[j] - arclengths[j-1]))
    end

    return grid, collect(τ)   
end

function mesh_region(region, ε::Function, band_index, T, n_levels::Int, n_cuts::Int, N = 1001, α = 6.0)
    n_levels = max(3, n_levels) # Enforce minimum of 2 patches in the energy direction
    n_cuts = max(3, n_cuts) # Enforce minimum of 2 patches in angular direction 

    ((x_min, x_max), (y_min, y_max)) = get_bounding_box(region)

    X = LinRange(x_min, x_max, N)
    Y = LinRange(y_min, y_max, N)
    E = Matrix{Float64}(undef, N, N) 
    for (i,x) in enumerate(X)
        for (j,y) in enumerate(Y)
            k = [x,y]
            if in_polygon(k, region)
                E[i,j] = ε(k)
            else
                E[i,j] = NaN # Identify point as living outside of IBZ
            end
        end
    end

    e_threshold = α*T # Half-width of Fermi tube
    e_min = max(-e_threshold, 0.999 * minimum(E[.!isnan.(E)]))
    e_max = min(e_threshold, 0.999 * maximum(E[.!isnan.(E)]))
    energies = collect(LinRange(e_min, e_max, 2 * n_levels - 1))
    # Shift energy levels to include corner energy if there is a crossing
    for corner_e ∈ ε.(region)
        if e_min < corner_e < e_max 
            i = argmin(abs.(energies .- corner_e))
            iseven(i) && (i = i + (-1)^argmin(abs.([energies[i - 1], energies[i + 1]] .- corner_e)) 
            )
            energies[i] = corner_e
        end
    end

    c = contours(X, Y, E, energies) # Generate Fermi surface contours

    patches = Matrix{Patch}(undef, n_levels-1, n_cuts-1)
    corners = Vector{SVector{2, Float64}}(undef, (2 * n_levels - 2) * n_cuts)
    corner_ids = Matrix{Vector{Int}}(undef, n_levels-1, n_cuts-1)

    cind = 1 # Corner iteration index

    for i in 2:2:2*n_levels-1
        k, arclengths = arclength_slice(c[i].isolines[1], 2 * n_cuts - 1)

        corners[cind], ij3 = contour_intersection(k[1], gradient(ε, k[1]), c[i-1].isolines[1])
        corners[cind+1], ij4 = contour_intersection(k[1], gradient(ε, k[1]), c[i+1].isolines[1])
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

            Δε = energies[i+1] - energies[i-1]
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
                ε(k[j]),
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

    return vec(patches)
end

mesh_region(region, ε, T, n_levels, n_cuts, N = 1001, α = 6.0) = mesh_region(region, ε, 1, T, n_levels, n_cuts, N, α)

function poly_area(poly, c)
    A = 0.0
    N = length(poly)
    oriented_poly = sort(poly, by = x -> atan(x[2] - c[2], x[1] - c[1]))

    for i in eachindex(oriented_poly)
        if i == N
            A += (oriented_poly[i][2] + oriented_poly[1][2]) * (oriented_poly[i][1] - oriented_poly[1][1])
        else
            A += (oriented_poly[i][2] + oriented_poly[i+1][2]) * (oriented_poly[i][1] - oriented_poly[i+1][1])
        end
    end
    return A/2
end

function ibz_mesh(l::Lattice, bands::Vector{Function}, T, n_levels::Int, n_cuts::Int, N::Int = 1001, α::Real = 6.0)
    full_patches = Vector{Patch}(undef, 0)
    ibz = get_ibz(l)

    for i in eachindex(bands)
        patches = mesh_region(ibz, bands[i], i, T, n_levels, n_cuts, N, α)
        full_patches = vcat(full_patches, patches)
    end

    return full_patches
end

ibz_mesh(l::Lattice, ε::Function, T, n_levels::Int, n_cuts::Int, N::Int = 1001, α::Real = 6.0) = ibz_mesh(l, [ε], T, n_levels, n_cuts, N, α)

function bz_mesh(l::Lattice, bands::Vector{Function}, T, n_levels::Int, n_cuts::Int, N::Int = 1001, α::Real = 6.0)
    G = point_group(l)
    ibz = get_ibz(l)

    θs = Vector{Float64}(undef, length(G.elements))
    for i in eachindex(G.elements)
        O = Groups.get_matrix_representation(G.elements[i])
        k = sum(map(x -> O*x, ibz)) # Central direction vector of IBZ
        θs[i] = mod(atan(k[2], k[1]), 2pi)
    end
    θperm = sortperm(θs)

    full_patches = Matrix{Patch}(undef, n_levels - 1, 0)
    # full_corners = Matrix{SVector{2,Float64}}(undef, n_levels, 0)

    for j in eachindex(bands)
        patches = mesh_region(ibz, bands[j], j, T, n_levels, n_cuts, N, α)

        size(patches)[2] == 0 && continue # No Fermi surface

        for i in θperm
            O = Groups.get_matrix_representation(G.elements[i])
            if det(O) < 0.0 # Improper rotation
                full_patches = hcat(full_patches, 
                    reverse( map(x -> patch_op(x, O), patches), dims = 2)
                )
            else
                full_patches = hcat(full_patches, 
                    map(x -> patch_op(x, O), patches) 
                )
            end
            # full_corners = hcat(full_corners, map(x -> O*x, corners))
        end
    end

    return full_patches
end

bz_mesh(l::Lattice, ε::Function, T, n_levels::Int, n_cuts::Int, N::Int = 1001, α::Real = 6.0) = bz_mesh(l, [ε], T, n_levels, n_cuts, N, α)

end