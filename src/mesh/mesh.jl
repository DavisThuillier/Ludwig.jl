export generate_mesh, Patch

struct Patch
    momentum::SVector{2,Float64}
    energy::Float64
    v::SVector{2,Float64}
    dV::Float64
    de::Float64
    jinv::Matrix{Float64} # Jacobian of transformation from (kx, ky) --> (E, θ)
    djinv::Float64 # Inverse jacobian determinant
    corners::Vector{Int} # Coordinates of corners for plotting
end

struct Mesh
    patches::Vector{Patch}
    corners::Vector{SVector{2, Float64}}
end

energy_threshold(T::Real, threshold::Real) = 2 * T * acosh(1 / (2 * sqrt(threshold)))

function angular_slice(iso::Isoline, n::Int)
    θ = LinRange(0.0, pi/2, n)

    curve = iso.points
    angles = map(get_angle, iso.points)

    grid = [curve[begin]]

    j = 1
    jmax = length(angles)
    for i in eachindex(θ)
        i == 1 && continue
        while j < jmax && angles[j] < θ[i]
            j += 1
        end
        j == 1 && @show curve[j]

        push!(grid, curve[j-1] + (curve[j] - curve[j - 1]) * (θ[i] - angles[j-1]) / (angles[j] - angles[j-1]))
    end

    return grid
end

function get_angle(k)
    if abs(k[2]) < 0.5 - abs(k[1])
        return mod(atan(k[2], k[1]), 2pi)
    else
        θ = atan(0.5 - abs(k[1]), 0.5 - abs(k[2]))
        if k[1] < 0.0
            if k[2] < 0.0
                return mod(θ - pi, 2pi)
            else
                return mod(θ + pi/2, 2pi)
            end
        else
            if k[2] < 0.0
                return mod(θ - pi/2, 2pi)
            else
                return mod(θ, 2pi) 
            end
        end
    end
end

function ∂θ∂x(k::AbstractVector)
    if abs(k[2]) < 0.5 - abs(k[1])
        return - k[2] / norm(k)^2
    else
        g = (0.5 .- abs.(k))
        return - sign(k[2]) * g[1] / norm(g)^2
    end
end

function ∂θ∂y(k::AbstractVector)
    if abs(k[2]) < 0.5 - abs(k[1])
        return k[1] / norm(k)^2
    else
        g = (0.5 .- abs.(k))
        return sign(k[1]) * g[2] / norm(g)^2
    end
end

function ∂θ∂k(k::AbstractVector)
    return SVector{2, Float64}([∂θ∂x(k), ∂θ∂y(k)])
end

function generate_mesh(h::Function, T::Real, n_levels::Int, n_angles::Int, N::Int = 1001, α::Real = 6.0)
    n_levels = max(3, n_levels) # Enforce minimum of 1 patches in the energy direction
    n_angles = max(3, n_angles) # Enforce minimum of 2 patches in angular direction
    Δθ = (pi/2) / (n_angles - 1) 

    corner_e = h([0.0, 0.5]) # Energy along diamond FS

    x = LinRange(0.0, 0.5, N)
    E = map(x -> h([x[1], x[2]]), collect(Iterators.product(x, x)))

    e_threshold = α*T # Buhmann's choice
    e_min = max(-e_threshold, 0.999 * minimum(E))
    e_max = min(e_threshold, 0.999 * maximum(E))
    Δε = (e_max - e_min) / (n_levels - 1)

    energies = collect(LinRange(e_min, e_max, n_levels))
    # if corner_e < e_max && corner_e > e_min
    #     mindex = argmin(abs.(energies .- corner_e))
    #     energies[mindex] = corner_e
    # end

    c = contours(x, x, E, energies)
    @show length(c[1].isolines)
    
    # Slice contours to generate patch corners
    corners = Matrix{SVector{2,Float64}}(undef, n_levels, n_angles)
    for i in eachindex(c)
        for curve in c[i].isolines
            first(curve.points)[1] < first(curve.points)[2] && reverse!(curve.points) # Enforce same curve orientation
            corners[i, :] = angular_slice(curve, n_angles)
        end
    end

    k = Matrix{SVector{2,Float64}}(undef, n_levels-1, n_angles-1)
    for i in 1:n_levels-1
        for j in 1:n_angles-1
            k[i,j] = (corners[i, j] + corners[i + 1, j] + corners[i + 1, j + 1] + corners[i, j + 1]) / 4
        end
    end
    v = map(x -> ForwardDiff.gradient(h, x), k)

    A = zeros(Float64, 2, 2) # Finite Difference Jacobian
    k1 = SVector{2, Float64}([0.0,0.0])
    J = zeros(Float64, 2, 2)
    patches = Matrix{Patch}(undef, n_levels-1, n_angles-1)
    for i in 1:size(patches)[1]
        for j in 1:size(patches)[2]
            if i == 1
                i2 = i+1; i1 = i # Forward difference
            elseif i == size(patches)[1]
                i2 = i; i1 = i-1 # Central difference
            else
                i2 = i+1; i1 = i-1 # Backward difference
            end 
            ΔE = h(k[i2,j]) - h(k[i1, j])
            A[1,1] = (k[i2,j][1] - k[i1,j][1]) / ΔE # ∂kx/∂ε |θ
            A[1,2] = (k[i2,j][2] - k[i1,j][2]) / ΔE # ∂ky/∂ε |θ

            if j == 1
                j2 = j+1; j1 = j
                Δϕ = get_angle(k[i,j2]) + get_angle(k[i,j1])
                if k[i, j1][2] < 0.5 - k[i, j1][1]
                    k1 = SVector{2}(k[i, j1][1], -k[i,j1][2]) # Quadrant IV patch
                else
                    k1 = SVector{2}(1.0 - k[i, j1][1], k[i,j1][2]) # Next BZ patch
                end 
            elseif j == size(patches)[2]
                j2 = j-1; j1 = j
                Δϕ = (get_angle(k[i,j2]) + get_angle(k[i,j1])) - pi 
                if k[i, j1][2] < 0.5 - k[i, j1][1]
                    k1 = SVector{2}(-k[i, j1][1], k[i,j1][2]) # Quadrant II patch
                else
                    k1 = SVector{2}(k[i, j1][1], 1.0 - k[i,j1][2]) # Next BZ patch
                end 
            else
                j2 = j+1; j1 = j-1
                Δϕ = (get_angle(k[i,j2]) - get_angle(k[i, j1]))
                k1 = k[i, j1]
            end 
            
            A[2,1] = (k[i,j2][1] - k1[1]) / Δϕ   # ∂ky/∂θ |ε
            A[2,2] = (k[i,j2][2] - k1[2]) / Δϕ   # ∂ky/∂θ |ε

            # Use forward differentiation to better calculate energy derivatives
            # J[1,1] = 2 * v[i,j][1] / Δε # ∂ε/∂kx |ky
            # J[1,2] = 2 * v[i,j][2] / Δε # ∂ε/∂ky |kx
            J[1,1] = 2 * v[i,j][1] / ΔE # ∂ε/∂kx |ky
            J[1,2] = 2 * v[i,j][2] / ΔE # ∂ε/∂ky |kx
            J[2,1] = - 2 * A[1,2] / det(A) / Δθ   # ∂θ/∂kx |ky
            J[2,2] = 2 * A[1,1] / det(A) / Δθ # ∂θ/∂ky |kx

            patches[i, j] = Patch(
                k[i,j], 
                h(k[i,j]),
                ForwardDiff.gradient(h, k[i,j]),
                get_patch_area(corners, i, j),
                abs(energies[i+1] - energies[i]),
                inv(J),
                1/det(J),
                [(j-1)*(n_levels) + i, (j-1)*(n_levels) + i+1, j*(n_levels) + i+1, j*(n_levels) + i]
            )
        end
    end

    # patches = hcat(patches,
    #     map(x -> rotate_patch(x, 1, length(corners)), patches),
    #     map(x -> rotate_patch(x, 2, 2*length(corners)), patches),
    #     map(x -> rotate_patch(x, 3, 3*length(corners)), patches)
    # )

    Mx = [1 0; 0 -1] # Mirror across the x-axis
    My = [-1 0; 0 1] # Mirror across the y-axis
    R = [0 -1; 1 0] # Rotation matrix for pi / 2

    patches = hcat(patches,
        map(x -> patch_op(x, My, mirror_perm, n_angles, n_levels, length(corners)), reverse(patches; dims = 2))
    )
    patches = hcat(patches,
        map(x -> patch_op(x, Mx, mirror_perm, 2*n_angles, n_levels, 2*length(corners)), reverse(patches; dims = 2))
    )

    corners = hcat(corners, 
        map(x -> My * x, reverse(corners; dims = 2))
    )
    corners = hcat(corners, 
        map(x -> Mx * x, reverse(corners; dims = (2)))
    )

    return Mesh(vec(patches), vec(corners)), Δε
end

function patch_op(x::Patch, M::Matrix, corner_perm::Function, n_col::Int, n_row::Int, corner_shift::Int)
    return Patch(
        SVector{2}(M * x.momentum), 
        x.energy,
        SVector{2}(M * x.v),
        x.dV,
        x.de,
        M * x.jinv, 
        x.djinv,
        corner_perm.(x.corners, n_col, n_row) .+ corner_shift
    )
end

function mirror_perm(i::Int, n_col::Int, n_row::Int)
    return Int( (n_col - ceil(i / n_row)) * n_row + mod(i - 1, n_row) + 1 )
end

function get_patch_area(A::Matrix, i::Int, j::Int)
    m = size(A)[2]
    if j != m
        α = A[i, j + 1] - A[i, j]
        β = A[i + 1, j + 1] - A[i, j + 1]
        γ = A[i + 1, j] - A[i + 1, j + 1]
    else
        α = A[i, 1] - A[i, j]
        β = A[i + 1, 1] - A[i, 1]
        γ = A[i + 1, j] - A[i + 1, 1]
    end
    δ = A[i, j] - A[i + 1, j]

    tri_area_1 = abs(α[1]*δ[2] - α[2]*δ[1]) / 2
    tri_area_2 = abs(β[1]*γ[2] - β[2]*γ[1]) / 2

    return tri_area_1 + tri_area_2
end