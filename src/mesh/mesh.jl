export multiband_mesh, generate_mesh, Patch

"""
Representation of a patch in momentum space to be integrated over when calculating the collision integral kernel.

# Fields
- `momentum`: Momentum in 1st BZ scaled by ``2\\pi / a``
- `energies`: Eigenvalues of Hamiltonian evaluated at `momentum`
- `band_index`: Index of the Fermi surface from which the patch was generated
- `v`: The group velocity at `momentum` taking `energies[band_index]` as the dispersion
- `dV`: Area of the patch in units of ``(2\\pi/a)^2``
- `jinv`: Jacobian of transformation from ``(k_x, k_y) \\mapsto (\\varepsilon, \\theta)``, the local patch coordinates
- `w`: The weights of the original orbital basis corresponding to the `band_index`th eigenvalue
- `corners`: Indices of coordinates in parent `Mesh` struct of corners for plotting
"""
struct Patch{D}
    momentum::SVector{2,Float64} # Momentum in 1st BZ
    energies::SVector{D, Float64} # Energy associated to band index and momentum
    band_index::Int 
    v::SVector{2,Float64} # Group velocity
    dV::Float64 # Patch area
    jinv::Matrix{Float64} # Jacobian of transformation from (kx, ky) --> (E, θ)
    djinv::Float64 # Inverse jacobian determinant
    w::SVector{D, Float64} # Weight vector of overlap with orbitals
    corners::Vector{Int} # Coordinates of corners for plotting
end


energy(p::Patch) = p.energies[p.band_index]

struct Mesh
    patches::Vector{Patch}
    corners::Vector{SVector{2, Float64}} # Corners of patches for plotting
    n_bands::Int
end

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

function multiband_mesh(H::Function, T::Real, n_levels::Int, n_angles::Int, N::Int = 1001, α::Real = 6.0)
    # Generate grid of 1st quadrant 
    x = LinRange(0.0, 0.5, N)
    
    n_bands = size(H([0.0, 0.0]))[1] # Number of bands
    E = Array{Float64}(undef, N, N, n_bands)
    for i in 1:N, j in 1:N
        E[i, j, :] .= eigvals(H([x[i], x[j]]))# Get eigenvalues (bands) of each k-point
    end
    # Here it is assumed that there is no level crossing, so that the bands remain ordered as the eigenvalues are calculated across the grid

    grid = Vector{Patch}(undef, 0)
    corners = Vector{SVector{2, Float64}}(undef, 0)
    
    Δε = 0.0
    for i in 1:n_bands
        mesh, Δε = generate_mesh(E[:,:,i], H, i, T, n_levels, n_angles, α)
        grid = vcat(grid, map(x -> Patch(
                                    x.momentum, 
                                    x.energies,
                                    x.v,
                                    x.dV,
                                    x.jinv, 
                                    x.djinv,
                                    x.corners .+ length(corners),
                                    x.band_index,
                                    x.w
                                ), mesh.patches)
        )
        corners = vcat(corners, mesh.corners)
    end

    return Mesh(grid, corners, n_bands), Δε

end

# Generate mesh for a single band given an array of values for the energy (output of diagonalizing hamiltonian)
function generate_mesh(E::AbstractArray{<:Real, 2}, H::Function, band_index::Int, T::Real, n_levels::Int, n_angles::Int, α::Real = 6.0)
    n_bands = size(H([0.0, 0.0]))[1] # Number of bands

    itp = interpolate(E, BSpline(Cubic(Line(OnGrid()))))
    x = LinRange(0.0, 0.5, size(E)[1])
    itp = scale(itp, x, x)

    n_levels = max(3, n_levels) # Enforce minimum of 1 patches in the energy direction
    n_angles = max(3, n_angles) # Enforce minimum of 2 patches in angular direction
    Δθ = (pi/2) / (n_angles - 1) 

    e_threshold = α*T # Half-width of Fermi tube
    e_min = max(-e_threshold, 0.999 * minimum(E))
    e_max = min(e_threshold, 0.999 * maximum(E))
    Δε = (e_max - e_min) / (n_levels - 1)

    energies = collect(LinRange(e_min, e_max, n_levels))
    
    x = LinRange(0.0, 0.5, size(E)[1])
    c = contours(x, x, E, energies) # Generate Fermi surface contours
    
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
            # Patch momentum is the arithmetic mean of
        end
    end
    v = map(x -> Interpolations.gradient(itp, x[1], x[2]), k)

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
            ΔE = itp(k[i2,j][1], k[i2,j][2]) - itp(k[i1,j][1], k[i1,j][2])
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
            
            A[2,1] = (k[i,j2][1] - k1[1]) / Δϕ   # ∂kx/∂θ |ε
            A[2,2] = (k[i,j2][2] - k1[2]) / Δϕ   # ∂ky/∂θ |ε

            # Use forward differentiation to better calculate energy derivatives
            J[1,1] = 2 * v[i,j][1] / ΔE # ∂ε/∂kx |ky
            J[1,2] = 2 * v[i,j][2] / ΔE # ∂ε/∂ky |kx
            J[2,1] = - 2 * A[1,2] / det(A) / Δθ   # ∂θ/∂kx |ky
            J[2,2] = 2 * A[1,1] / det(A) / Δθ # ∂θ/∂ky |kx

            ## Get weights from transformation matrix ##
            w::SVector{n_bands, Float64} = eigvecs(H(k[i,j]))[:, band_index]
            e::SVector{n_bands, Float64} = eigvals(H(k[i,j]))

            patches[i, j] = Patch(
                k[i,j], 
                e,
                Interpolations.gradient(itp, k[i,j][1], k[i,j][2]),
                get_patch_area(corners, i, j),
                inv(J),
                1/det(J),
                [(j-1)*(n_levels) + i, (j-1)*(n_levels) + i+1, j*(n_levels) + i+1, j*(n_levels) + i],
                band_index,
                w
            )
        end
    end
    
    Mx = [1 0; 0 -1] # Mirror across the x-axis
    My = [-1 0; 0 1] # Mirror across the y-axis
    R = [0 -1; 1 0] # Rotation matrix for pi / 2

    # patches = hcat(patches,
    #     map(x -> rotate_patch(x, 1, length(corners)), patches),
    #     map(x -> rotate_patch(x, 2, 2*length(corners)), patches),
    #     map(x -> rotate_patch(x, 3, 3*length(corners)), patches)
    # )

    # corners = hcat(corners,
    # map(x -> R * x, corners),
    # map(x -> R^2 * x, corners),
    # map(x -> R^3 * x, corners)
    # )


    patches = vcat(patches,
        map(x -> patch_op(x, My, mirror_perm, n_angles, n_levels, length(corners)), patches)
    )
    patches = vcat(patches,
        map(x -> patch_op(x, Mx, mirror_perm, 2 * n_angles, n_levels, 2 * length(corners)), patches)
    )


    corners = hcat(corners,
    map(x -> R * x, corners),
    map(x -> R^2 * x, corners),
    map(x -> R^3 * x, corners)
    )

    
    return Mesh(vec(patches), vec(corners), n_bands), Δε
end

# Generates mesh for a single band given a functional form of the dispersion
function generate_mesh(h::Function, T::Real, n_levels::Int, n_angles::Int, N::Int = 1001, α::Real = 6.0)
    n_levels = max(3, n_levels) # Enforce minimum of 1 patches in the energy direction
    n_angles = max(3, n_angles) # Enforce minimum of 2 patches in angular direction
    Δθ = (pi/2) / (n_angles - 1) 

    # Generate grid of 1st quadrant 
    x = LinRange(0.0, 0.5, N)
    E = map(x -> h([x[1], x[2]]), collect(Iterators.product(x, x)))

    e_threshold = α*T # Half-width of Fermi tube
    e_min = max(-e_threshold, 0.999 * minimum(E))
    e_max = min(e_threshold, 0.999 * maximum(E))
    Δε = (e_max - e_min) / (n_levels - 1)

    energies = collect(LinRange(e_min, e_max, n_levels))

    c = contours(x, x, E, energies) # Generate Fermi surface contours
    
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
            # Patch momentum is the arithmetic mean of
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
            
            A[2,1] = (k[i,j2][1] - k1[1]) / Δϕ   # ∂kx/∂θ |ε
            A[2,2] = (k[i,j2][2] - k1[2]) / Δϕ   # ∂ky/∂θ |ε

            # Use forward differentiation to better calculate energy derivatives
            J[1,1] = 2 * v[i,j][1] / ΔE # ∂ε/∂kx |ky
            J[1,2] = 2 * v[i,j][2] / ΔE # ∂ε/∂ky |kx
            J[2,1] = - 2 * A[1,2] / det(A) / Δθ   # ∂θ/∂kx |ky
            J[2,2] = 2 * A[1,1] / det(A) / Δθ # ∂θ/∂ky |kx

            patches[i, j] = Patch(
                k[i,j], 
                h(k[i,j]),
                ForwardDiff.gradient(h, k[i,j]),
                get_patch_area(corners, i, j),
                inv(J),
                1/det(J),
                [(j-1)*(n_levels) + i, (j-1)*(n_levels) + i+1, j*(n_levels) + i+1, j*(n_levels) + i],
                1, # Single band
                [1] # Single band transformation matrix
            )
        end
    end

    Mx = [1 0; 0 -1] # Mirror across the x-axis
    My = [-1 0; 0 1] # Mirror across the y-axis
    R = [0 -1; 1 0] # Rotation matrix for pi / 2

    patches = hcat(patches,
        map(x -> rotate_patch(x, 1, length(corners)), patches),
        map(x -> rotate_patch(x, 2, 2*length(corners)), patches),
        map(x -> rotate_patch(x, 3, 3*length(corners)), patches)
    )

    corners = hcat(corners,
        map(x -> R * x, corners),
        map(x -> R^2 * x, corners),
        map(x -> R^3 * x, corners)
    )

    return Mesh(vec(patches), vec(corners), 1), Δε
end

function patch_op(p::Patch, M::Matrix, corner_perm::Function, n_col::Int, n_row::Int, corner_shift::Int)
    return Patch(
        SVector{2}(M * p.momentum), 
        p.energy,
        SVector{2}(M * p.v),
        p.dV,
        M * p.jinv, 
        p.djinv,
        corner_perm.(p.corners, n_col, n_row) .+ corner_shift,
        p.band_index,
        p.w
    )
end

function rotate_patch(p, n, shift)
    R = [0 -1; 1 0]^n
    return Patch(
        SVector{2}(R * p.momentum),
        p.energy,
        SVector{2}(R * p.v),
        p.dV,
        R * p.jinv,
        p.djinv,
        p.corners .+ shift, 
        p.band_index,
        p.w
    )
end

function mirror_perm(i::Int, n_col::Int, n_row::Int)
    return Int( (n_col - ceil(i / n_row)) * n_row + mod(i - 1, n_row) + 1 )
end

"""
    get_patch_area(A, i, j)

Determine the area of the quadrilateral bounded by `A[i,j]`, `A[i + 1, j]`, `A[i + 1, j + 1]`, and `A[i, j+1]`.
"""
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