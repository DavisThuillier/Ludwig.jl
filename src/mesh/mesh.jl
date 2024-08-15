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
    energy::Float64 # Energy associated to band index and momentum
    bounds::SVector{2,Float64}
    band_index::Int 
    v::SVector{2,Float64} # Group velocity
    dV::Float64 # Patch area
    de::Float64
    jinv::Matrix{Float64} # Jacobian of transformation from (kx, ky) --> (E, θ)
    djinv::Float64 # Absolute value of inverse jacobian determinant
    w::SVector{D, Float64} # Weight vector of overlap with orbitals
    corners::Vector{Int} # Coordinates of corners for plotting
end

"""
    energy(p::Patch)

Return the energy corresponding to the band for which `p` was generated.
"""
energy(p::Patch) = p.energy

"""
Container struct for patches over which to integrate.

# Fields
- `patches`: Vector of patches
- `corners`: Vector of points on patch corners for plotting mesh
- `n_bands`: Dimension of the generating Hamiltonian
"""
struct Mesh
    patches::Vector{Patch}
    corners::Vector{SVector{2, Float64}} # Corners of patches for plotting
    n_bands::Int
    α # Width of Fermi tube is 2α
end

"""
    angular_slice(iso, n)

Generate a vector of `n` points equally distributed in angle along `iso`. 
"""
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

"""
    get_angle(k)

Return the angle defined with respect to ``(0,0)`` if `k` is within the umklapp surface and with respect to ``(\\pi/2, \\pi/2)`` otherwise.
"""
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

"""
    multiband_mesh(bands, W, T, n_levels, n_angles[, N, α])

Generate a Mesh of (`n_angles` - 1) x (`n_levels` - 1) patches per quadrant per band centered on each band's thermally broadened Fermi surface at temperature `T`.

`bands` is a vector of the . The width of the Fermi tube at each surface is ``\\pm`` `α T`. 
"""
function multiband_mesh(bands::Vector, W::Function, T::Real, n_levels::Int, n_angles::Int; N::Int = 2001, α::Real = 6.0)
    grid = Vector{Patch}(undef, 0)
    corners = Vector{SVector{2, Float64}}(undef, 0)
    n_bands = length(bands)
    
    for i in eachindex(bands)
        mesh = generate_mesh(bands, W, i, n_bands, T, n_levels, n_angles, N, α)
        grid = vcat(grid, map(x -> Patch(
                                    x.momentum, 
                                    x.energy,
                                    x.bounds,
                                    x.band_index,
                                    x.v,
                                    x.dV,
                                    x.de,
                                    x.jinv, 
                                    x.djinv,
                                    x.w,
                                    x.corners .+ length(corners)
                                ), mesh.patches)
        )
        corners = vcat(corners, mesh.corners)
    end

    return Mesh(grid, corners, n_bands, α)

end

"""
    generate_mesh(E::AbstractArray{<:Real, 2}, H, band_index, T, n_levels, n_angles[, α])

Generate a `Mesh` of (`n_angles` - 1) x (`n_levels` - 1) patches per quadrant using marching squares to find energy contours of `E`, the energy eigenvalues of `H` corresponding to `band_index` evaluated on a regular grid of ``\\mathbf{k} \\in [0.0, 0.5]\\times[0.0, 0.5]``. 

The width of the Fermi tube is ``\\pm`` `α T`. 
"""
function generate_mesh(bands, W::Function, band_index::Int, n_bands::Int, T::Real, n_levels::Int, n_angles::Int, N::Int = 1001, α::Real = 6.0)
    n_levels = max(3, n_levels) # Enforce minimum of 1 patches in the energy direction
    n_angles = max(3, n_angles) # Enforce minimum of 2 patches in angular direction
    Δθ = (pi/2) / (n_angles - 1) 

    x = LinRange(0.0, 0.5, N)
    E = map(x -> bands[band_index]([x[1], x[2]]), collect(Iterators.product(x, x)))

    e_threshold = α*T # Half-width of Fermi tube
    e_min = max(-e_threshold, 0.999 * minimum(E))
    e_max = min(e_threshold, 0.999 * maximum(E))
    Δε = (e_max - e_min) / (n_levels - 1)
    energies = collect(LinRange(e_min, e_max, 2 * n_levels - 1))

    c = contours(x, x, E, energies) # Generate Fermi surface contours
    
    # Slice contours to generate patch corners
    corners = Matrix{SVector{2,Float64}}(undef, n_levels, n_angles)
    k = Matrix{SVector{2,Float64}}(undef, n_levels-1, n_angles-1)
    for i in eachindex(c)
        for curve in c[i].isolines
            first(curve.points)[1] < first(curve.points)[2] && reverse!(curve.points) # Enforce same curve orientation
            if isodd(i)
                corners[i ÷ 2 + 1, :] = angular_slice(curve, n_angles)
            else
                k[i ÷ 2, :] = angular_slice(curve, 2 * n_angles - 1)[2:2:end]
            end
        end
    end

    v = map(x -> ForwardDiff.gradient(bands[band_index], x), k)

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
            ΔE = bands[band_index](k[i2,j]) - bands[band_index](k[i1,j])
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
            w::SVector{n_bands, Float64} = W(k[i,j])[:, band_index]

            patches[i, j] = Patch(
                k[i,j], 
                bands[band_index](k[i,j]),
                SVector{2}(energies[2i - 1], energies[2i + 1]),
                band_index,
                v[i,j],
                get_patch_area(corners, i, j),
                Δε,
                inv(J),
                1/abs(det(J)), 
                w,
                [(j-1)*(n_levels) + i, (j-1)*(n_levels) + i+1, j*(n_levels) + i+1, j*(n_levels) + i]
            )
        end
    end
    
    Mx = [1 0; 0 -1] # Mirror across the x-axis
    My = [-1 0; 0 1] # Mirror across the y-axis
    R = [0 -1; 1 0] # Rotation matrix for pi / 2

    patches = hcat(patches,
        reverse(
            map(x -> patch_op(x, My, mirror_perm, n_angles, n_levels, length(corners)), patches), dims=2
        )
    )
    patches = hcat(patches,
        reverse(
            map(x -> patch_op(x, Mx, mirror_perm, 2 * n_angles, n_levels, 2 * length(corners)), patches), dims=2
        )
    )

    corners = hcat(corners,
    map(x -> R * x, corners),
    map(x -> R^2 * x, corners),
    map(x -> R^3 * x, corners)
    )

    return Mesh(vec(patches), vec(corners), n_bands, α)
end

generate_mesh(ε::Function, T::Real, n_levels::Int, n_angles::Int, N::Int = 1001, α::Real = 6.0) = generate_mesh([ε], x -> [1.0], 1, 1, T, n_levels, n_angles, N, α)

function get_patch_momentum(corners, i, j, ε, ε0)
    u = corners[i + 1, j] - corners[i, j]
    v = corners[i, j + 1] - corners[i, j]

    x = LinRange(0.0, 1.0, 4)
    k = map(x -> corners[i,j] + x[1] * u + x[2] * v, collect(Iterators.product(x, x)))
    E = map(x -> ε(corners[i,j] + x[1] * u + x[2] * v), collect(Iterators.product(x, x)))
    c = find_contour(x, x, E, ε0).isolines
    
    if length(c) == 0
        display(k)
        @show corners[i + 1, j + 1]
        @show ε0
    end

    iso = c[1]
    k = 1
    s = 0.0
    while k < length(iso.points) && s < iso.arclength / 2.0
        k += 1
        s += norm(iso.points[k] - iso.points[k - 1])
    end

    coords = iso.points[k - 1] + (iso.points[k] - iso.points[k-1]) * (s - iso.arclength/2.0) / norm((iso.points[k] - iso.points[k-1])) # Coordinates in u,v space

    return corners[i,j] + coords[1] * u + coords[2] * v
end

function patch_op(p::Patch, M::Matrix, corner_perm::Function, n_col::Int, n_row::Int, corner_shift::Int)
    return Patch(
        SVector{2}(M * p.momentum), 
        p.energy,
        p.bounds,
        p.band_index,
        SVector{2}(M * p.v),
        p.dV,
        p.de,
        M * p.jinv, 
        p.djinv,
        p.w,
        corner_perm.(p.corners, n_col, n_row) .+ corner_shift,
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