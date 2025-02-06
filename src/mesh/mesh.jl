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
struct Patch{D, T}
    momentum::SVector{2, T} # Momentum in 1st BZ
    energy::T # Energy associated to band index and momentum
    band_index::Int 
    v::SVector{2,T} # Group velocity
    dV::T # Patch area
    de::T 
    jinv::Matrix{T} # Jacobian of transformation from (kx, ky) --> (E, θ)
    djinv::T # Absolute value of inverse jacobian determinant
    w::SVector{D} # Weight vector of overlap with orbitals
    corners::Vector{Int} # Coordinates of corners for plotting
end

"""
Container struct for patches over which to integrate.

# Fields
- `patches`: Vector of patches
- `corners`: Vector of points on patch corners for plotting mesh
- `n_bands`: Dimension of the generating Hamiltonian
- `α`: Half the width of the Fermi tube
"""
struct Mesh
    patches::Vector{Patch}
    corners::Vector{SVector{2, Float64}} 
    n_bands::Int
    α
end

function Base.iterate(m::Mesh, state=0)
    state > nfields(m) && return
    return Base.getfield(m, state+1), state+1
end

"""
    angular_slice(iso, n)

Generate a vector of `n` points equally distributed in angle along `iso`. 
"""
function angular_slice(iso::MarchingSquares.Isoline, n::Int)
    curve = iso.points
    angles = map(get_angle, iso.points)

    θ = LinRange(minimum(angles), maximum(angles), n)

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

`bands` is a vector of functions defining the eigenvalues of the single-electron Hamiltonian over the 1st Brillouin Zone. The function `W` is an array-valued function whose input is a momentum 2-vector, and whose output columns are the the eigenvectors corresponding to `bands`. 
The width of the Fermi tube at each surface is ``\\pm`` `α T`. 
"""
function multiband_mesh(bands::Vector, orbital_weights::Function, T::Real, n_levels::Int, n_angles::Int; N::Int = 2001, α::Real = 6.0)
    grid = Vector{Patch}(undef, 0)
    corners = Vector{SVector{2, Float64}}(undef, 0)
    n_bands = length(bands)
    
    for i in eachindex(bands)
        generate_mesh(bands, orbital_weights, i, n_bands, T, n_levels, n_angles, N, α, length(corners)) |> 
        x -> begin 
            append!(grid, x.patches)
            append!(corners, x.corners)
        end
    end

    return Mesh(grid, corners, n_bands, α)

end

"""
    generate_mesh(bands, W::Function, band_index::Int, n_bands::Int, T::Real, n_levels::Int, n_angles::Int)

Generate a single-band `Mesh` in a multiband system; the mesh will have (`n_angles` - 1) x (`n_levels` - 1) patches per quadrant centered at the Fermi surface using marching squares to find energy contours of `bands[band_index]` evaluated on a regular grid of ``\\mathbf{k} \\in [0.0, 0.5]\\times[0.0, 0.5]``. The width of the Fermi tube is ``\\pm`` `α T`. 

The function `orbital_weights` is an array-valued function whose input is a momentum 2-vector, and whose output columns are the the eigenvectors of the single-electron Hamiltonian whose eigenvalues are the functions in `bands`.
"""
function generate_mesh(bands, orbital_weights::Function, band_index::Int, n_bands::Int, T::Real, n_levels::Int, n_angles::Int, N::Int, α::Real, band_corner_shift::Int = 0)
    n_levels = max(3, n_levels) # Enforce minimum of 2 patches in the energy direction
    n_angles = max(3, n_angles) # Enforce minimum of 2 patches in angular direction
    Δθ = (pi/2) / (n_angles - 1) 

    x = LinRange(0.0, 0.5, N)
    E = map(x -> bands[band_index]([x[1], x[2]]), collect(Iterators.product(x, x)))

    umklapp_edge_energy = bands[band_index]([0.0, 0.5])

    e_threshold = α*T # Half-width of Fermi tube
    e_min = max(-e_threshold, 0.999 * minimum(E))
    e_max = min(e_threshold, 0.999 * maximum(E))
    Δε = (e_max - e_min) / (n_levels - 1)

    energies = collect(LinRange(e_min, e_max, 2 * n_levels - 1))

    if e_min < umklapp_edge_energy < e_max 
        i = argmin(abs.(energies .- umklapp_edge_energy))
        iseven(i) && (i = i + (-1)^argmin(abs.([energies[i - 1], energies[i + 1]] .- umklapp_edge_energy)) 
        )
        energies[i] = umklapp_edge_energy
    end

    c = MarchingSquares.contours(x, x, E, energies) # Generate Fermi surface contours
    
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
            w::SVector{n_bands, Float64} = orbital_weights(k[i,j])[:, band_index]

            patches[i, j] = Patch(
                k[i,j], 
                bands[band_index](k[i,j]),
                band_index,
                v[i,j],
                get_patch_area(corners, i, j),
                Δε,
                inv(J),
                1/abs(det(J)), 
                w,
                [(j-1)*(n_levels) + i, (j-1)*(n_levels) + i+1, j*(n_levels) + i+1, j*(n_levels) + i] .+ band_corner_shift
            )
        end
    end
    
    Mx = [1 0; 0 -1] # Mirror across the x-axis
    My = [-1 0; 0 1] # Mirror across the y-axis

    patches = hcat(patches,
        reverse(
            map(x -> patch_op(x, My, length(corners)), patches), dims=2
        )
    )
    patches = hcat(patches,
        reverse(
            map(x -> patch_op(x, Mx, 2 * length(corners)), patches), dims=2
        )
    )

    corners = hcat(corners,
    map(x -> My * x, corners),
    )
    corners = hcat(corners, map(x -> Mx * x, corners))

    return Mesh(vec(patches), vec(corners), n_bands, α)
end

"""
    generate_mesh(ε::Function, T::Real, n_levels::Int, n_angles::Int[, N, α])

Generate a `Mesh` of (`n_angles` - 1) x (`n_levels` - 1) patches per quadrant centered at the Fermi surface using marching squares to find energy contours of `ε` evaluated on a regular grid of ``\\mathbf{k} \\in [0.0, 0.5]\\times[0.0, 0.5]``. The width of the Fermi tube is ``\\pm`` `α T`. 
"""
generate_mesh(ε::Function, T::Real, n_levels::Int, n_angles::Int, N::Int = 2001, α::Real = 6.0) = generate_mesh([ε], x -> [1.0], 1, 1, T, n_levels, n_angles, N, α)

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

function arclength_slice(iso::MarchingSquares.Isoline, n::Int)
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

function _ibz_mesh(l::Lattices.Lattice, bands, orbital_weights, band_index::Int, T, n_levels::Int, n_cuts::Int, N::Int = 1001, α::Real = 6.0, band_corner_shift::Int = 0)
    n_levels = max(3, n_levels) # Enforce minimum of 2 patches in the energy direction
    n_cuts = max(3, n_cuts) # Enforce minimum of 2 patches in angular direction 
    n_bands = length(bands)

    ibz = Lattices.get_ibz(l)

    a = sort(ibz, by = norm)[2]
    ϕa = atan(a[2], a[1]) # Reference angle

    X = LinRange(minimum(first.(ibz)), maximum(first.(ibz)), N)
    Y = LinRange(minimum(last.(ibz)), maximum(last.(ibz)), N)

    E = Matrix{Float64}(undef, N, N) 
    for (i,x) in enumerate(X)
        for (j,y) in enumerate(Y)
            k = [x,y]
            if Lattices.in_polygon(k, ibz)
                E[i,j] = bands[band_index](k)
            else
                E[i,j] = NaN # Identify point as living outside of IBZ
            end
        end
    end

    e_threshold = α*T # Half-width of Fermi tube
    e_min = max(-e_threshold, 0.999 * minimum(E[.!isnan.(E)]))
    e_max = min(e_threshold, 0.999 * maximum(E[.!isnan.(E)]))

    if e_min >= 0.0 || e_max <= 0.0
        return Matrix{Patch}(undef, n_levels - 1, 0), Matrix{SVector{2, Float64}}(undef, n_levels, 0)
    end

    Δε = (e_max - e_min) / (n_levels - 1)
    
    energies = collect(LinRange(e_min, e_max, 2 * n_levels - 1))

    # Shift energy levels to include corner energy if there is a crossing
    for k in ibz
        corner_e = bands[band_index](k)
        if e_min < corner_e < e_max 
            i = argmin(abs.(energies .- corner_e))
            iseven(i) && (i = i + (-1)^argmin(abs.([energies[i - 1], energies[i + 1]] .- corner_e)) 
            )
            energies[i] = corner_e
        end
    end

    c = MarchingSquares.contours(X, Y, E, energies) # Generate Fermi surface contours
    
    # Slice contours to generate patch corners
    corners = Matrix{SVector{2,Float64}}(undef, n_levels, n_cuts)
    corner_arclengths = Matrix{Float64}(undef, n_levels, n_cuts)
    k = Matrix{SVector{2,Float64}}(undef, n_levels-1, n_cuts-1)
    k_arclengths = Matrix{Float64}(undef, n_levels-1, 2 * n_cuts-1)
    
    Δs = Vector{Float64}(undef, n_levels-1)
    for i in eachindex(c)
        curve = c[i].isolines[1]

        ϕ1 = atan(curve.points[begin][2], curve.points[begin][1]) # Angle of start of curve
        ϕ2 = atan(curve.points[end][2], curve.points[end][1]) # Angle of end of curve
        if abs(mod(ϕ2 - ϕa, 2pi)) < abs(mod(ϕ1 - ϕa, 2pi))
            reverse!(curve.points)
        end

        if isodd(i)
            corners[i ÷ 2 + 1, :], corner_arclengths[i ÷ 2 + 1, :] = arclength_slice(curve, n_cuts)
        else
            temp_k, k_arclengths[i ÷ 2, :] = arclength_slice(curve, 2 * n_cuts - 1)
            k[i ÷ 2, :] = temp_k[2:2:end]
            Δs[i ÷ 2] = (k_arclengths[i ÷ 2, end] - k_arclengths[i ÷ 2, begin]) / (n_cuts - 1)
        end
    end

    v = map(x -> ForwardDiff.gradient(bands[band_index], x), k)

    A = zeros(Float64, 2, 2) # Finite Difference Jacobian
    k1 = SVector{2, Float64}([0.0,0.0])
    J = zeros(Float64, 2, 2)
    patches = Matrix{Patch}(undef, n_levels-1, n_cuts-1)
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
            elseif j == size(patches)[2]
                j2 = j-1; j1 = j
            else
                j2 = j+1; j1 = j-1
            end 
            Δϕ = k_arclengths[i, 2*j2] - k_arclengths[i, 2*j1]
            k1 = k[i,j1]
            
            A[2,1] = (k[i,j2][1] - k1[1]) / Δϕ   # ∂kx/∂s |ε
            A[2,2] = (k[i,j2][2] - k1[2]) / Δϕ   # ∂ky/∂s |ε

            # Use forward differentiation to better calculate energy derivatives
            J[1,1] = 2 * v[i,j][1] / ΔE # ∂ε/∂kx |ky
            J[1,2] = 2 * v[i,j][2] / ΔE # ∂ε/∂ky |kx
            J[2,1] = - 2 * A[1,2] / det(A) / Δs[i]   # ∂s/∂kx |ky
            J[2,2] = 2 * A[1,1] / det(A) / Δs[i] # ∂s/∂ky |kx

            w = orbital_weights(k[i,j])[:, band_index]
            w = SVector{length(w), ComplexF64}(w)

            patches[i, j] = Patch(
                k[i,j], 
                bands[band_index](k[i,j]),
                band_index,
                v[i,j],
                get_patch_area(corners, i, j),
                Δε,
                inv(J),
                1/abs(det(J)), 
                w,
                [(j-1)*(n_levels) + i, (j-1)*(n_levels) + i+1, j*(n_levels) + i+1, j*(n_levels) + i] .+ band_corner_shift
            )
        end
    end

    return patches, corners
end

function generate_ibz_mesh(l::Lattices.Lattice, bands::Vector{Function}, orbital_weights, T, n_levels::Int, n_cuts::Int, N::Int = 1001, α::Real = 6.0)
    grid = Vector{Patch}(undef, 0)
    corners = Vector{SVector{2, Float64}}(undef, 0)
    n_bands = length(bands)
    
    for i in eachindex(bands)
        _ibz_mesh(l, bands, orbital_weights, i, T, n_levels, n_cuts, N, α, length(corners)) |> 
        x -> begin 
            append!(grid, x[1])
            append!(corners, x[2])
        end
    end

    return Mesh(grid, corners, n_bands, α)

end

function generate_bz_mesh(l::Lattices.Lattice, bands, orbital_weights, T, n_levels::Int, n_cuts::Int, N::Int = 1001, α::Real = 6.0)
    G = Lattices.point_group(l)
    ibz = Lattices.get_ibz(l)

    θs = Vector{Float64}(undef, length(G.elements))
    for i in eachindex(G.elements)
        O = Groups.get_matrix_representation(G.elements[i])
        k = sum(map(x -> O*x, ibz)) # Central direction vector of IBZ
        θs[i] = mod(atan(k[2], k[1]), 2pi)
    end
    θperm = sortperm(θs)

    full_patches = Matrix{Patch}(undef, n_levels - 1, 0)
    full_corners = Matrix{SVector{2,Float64}}(undef, n_levels, 0)

    for j in eachindex(bands)
        patches, corners = _ibz_mesh(l, bands, orbital_weights, j, T, n_levels, n_cuts, N, α)

        size(patches)[2] == 0 && continue # No Fermi surface

        for i in θperm
            O = Groups.get_matrix_representation(G.elements[i])
            if det(O) < 0.0 # Improper rotation
                full_patches = hcat(full_patches, 
                    reverse( map(x -> patch_op(x, O, length(full_corners)), patches), dims = 2)
                )
            else
                full_patches = hcat(full_patches, 
                    map(x -> patch_op(x, O, length(full_corners)), patches) 
                )
            end
            full_corners = hcat(full_corners, map(x -> O*x, corners))
        end
    end

    return Mesh(vec(full_patches), vec(full_corners), length(bands), α)
end

"""
    patch_op(p, M, corner_shift)

Perform the symmetry operation of `M` on patch `p` to generate another patch. 

`corner_perm` is a permutation function that takes `n_col`, `n_row` and `p.corners` as input to determine the ids of the patch corners in a corresponding array. This is for plotting purposes only. 
"""
function patch_op(p::Patch, M::Matrix, corner_shift::Int)
    return Patch(
        SVector{2}(M * p.momentum), 
        p.energy,
        p.band_index,
        SVector{2}(M * p.v),
        p.dV,
        p.de,
        M * p.jinv, 
        p.djinv,
        p.w,
        p.corners .+ corner_shift
    )
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
