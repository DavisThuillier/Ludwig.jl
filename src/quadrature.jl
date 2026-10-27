###
### Discrete Laplacian
###

function discrete_laplacian(mesh::SimplexMesh{D}) where {D}
    N = length(mesh.vertices)
    S = length(mesh.simplices)

    nonzero_count = 4S * (2D - 3) # Number of nonzero elements per simplex x number of simplices

    row_indices = Vector{Int}(undef, nonzero_count)
    column_indices = Vector{Int}(undef, nonzero_count)
    nonzero_values = Vector{Float64}(undef, nonzero_count)

    for s ∈ 1:S
        populate_sparse_laplacian!(row_indices, column_indices, nonzero_values, s, mesh)
    end

    return sparse(row_indices, column_indices, nonzero_values, N, N)
end

function populate_sparse_laplacian!(row_indices, column_indices, nonzero_values, s::Int, mesh::SimplexMesh{2})
    offset = 4 * (s - 1)
    i, j = mesh.simplices[s]
    simplex_measure = volume(s, mesh)
    edge_weight = 1 / simplex_measure
    row_indices[offset+1] = i; column_indices[offset+1] = i; nonzero_values[offset+1] = edge_weight
    row_indices[offset+2] = j; column_indices[offset+2] = j; nonzero_values[offset+2] = edge_weight
    row_indices[offset+3] = i; column_indices[offset+3] = j; nonzero_values[offset+3] = -edge_weight
    row_indices[offset+4] = j; column_indices[offset+4] = i; nonzero_values[offset+4] = -edge_weight
end

function populate_sparse_laplacian!(row_indices, column_indices, nonzero_values, s::Int, mesh::SimplexMesh{3})
    offset = 12 * (s - 1)
    i, j, k = mesh.simplices[s]
    pᵢ, pⱼ, pₖ = mesh.vertices[i], mesh.vertices[j], mesh.vertices[k]
    simplex_measure = volume(s, mesh)

    lij² = sum(abs2, pⱼ - pᵢ)
    lik² = sum(abs2, pₖ - pᵢ)
    ljk² = sum(abs2, pₖ - pⱼ)

    w_ij = (lik² + ljk² - lij²) / (8simplex_measure) # angle at k, opposite edge i–j
    w_jk = (lij² + lik² - ljk²) / (8simplex_measure) # angle at i, opposite edge j–k
    w_ik = (lij² + ljk² - lik²) / (8simplex_measure) # angle at j, opposite edge i–k

    row_indices[offset+1] = i; column_indices[offset+1] = i; nonzero_values[offset+1] = w_ij
    row_indices[offset+2] = j; column_indices[offset+2] = j; nonzero_values[offset+2] = w_ij
    row_indices[offset+3] = i; column_indices[offset+3] = j; nonzero_values[offset+3] = -w_ij
    row_indices[offset+4] = j; column_indices[offset+4] = i; nonzero_values[offset+4] = -w_ij

    row_indices[offset+5] = j; column_indices[offset+5] = j; nonzero_values[offset+5] = w_jk
    row_indices[offset+6] = k; column_indices[offset+6] = k; nonzero_values[offset+6] = w_jk
    row_indices[offset+7] = j; column_indices[offset+7] = k; nonzero_values[offset+7] = -w_jk
    row_indices[offset+8] = k; column_indices[offset+8] = j; nonzero_values[offset+8] = -w_jk

    row_indices[offset+9]  = i; column_indices[offset+9]  = i; nonzero_values[offset+9]  = w_ik
    row_indices[offset+10] = k; column_indices[offset+10] = k; nonzero_values[offset+10] = w_ik
    row_indices[offset+11] = i; column_indices[offset+11] = k; nonzero_values[offset+11] = -w_ik
    row_indices[offset+12] = k; column_indices[offset+12] = i; nonzero_values[offset+12] = -w_ik
end

# 2D Laplacians are positive semidefinite, so Cholesky catches a loss of definiteness;
# 3D cotangent Laplacians can be indefinite, so LDLᵀ is required.
factorize_laplacian(A, σ, ::SimplexMesh{2}) = cholesky(Symmetric(A); shift = σ)
factorize_laplacian(A, σ, ::SimplexMesh{3}) = ldlt(Symmetric(A); shift = σ)

function mass_matrix(mesh::SimplexMesh{D}, ρ::Function) where {D}
    M = zeros(Float64, length(mesh.vertices))
    for s ∈ eachindex(mesh.simplices)
        simplex_measure = volume(s, mesh)
        for k ∈ mesh.simplices[s]
            M[k] += simplex_measure / D * ρ(mesh.vertices[k])
        end
    end
    return M
end

"""
    mesh_harmonics(mesh::SimplexMesh, ρ::Function, N; σ, maxiter)
    mesh_harmonics(mesh::SimplexMesh, M::AbstractVector, N; σ, maxiter)

The first `N` harmonics of the measure `dμ = ρ dS` on `mesh`, returned as the columns of a matrix: the lowest modes of the generalized eigenproblem `Lϕ = λMϕ` for the discrete Laplacian `L` and mass matrix `M`. The mass may be given as a density `ρ` lumped on `mesh`, or supplied directly as a vertex-mass vector `M` so a caller can inform it from a finer integration mesh. `σ` regularizes the shift-invert factorization; `maxiter` bounds the eigensolver iterations.
"""
mesh_harmonics(mesh::SimplexMesh, ρ::Function, N::Int; kwargs...) =
    mesh_harmonics(mesh, mass_matrix(mesh, ρ), N; kwargs...)

function mesh_harmonics(mesh::SimplexMesh, M::AbstractVector, N::Int; σ::Real = 1e-16, maxiter = 100)
    length(M) == length(mesh.vertices) ||
        throw(DimensionMismatch("mass vector has $(length(M)) entries but mesh has $(length(mesh.vertices)) vertices"))
    N_modes = min(N, length(mesh.vertices) - 1)
    L = discrete_laplacian(mesh)
    inv_sqrt_mass = Diagonal(1 ./ sqrt.(M))
    normalized_laplacian = inv_sqrt_mass * L * inv_sqrt_mass
    # A uniform scalar rescale leaves the eigenvectors, and hence the harmonics, unchanged.
    scale = maximum(abs, diag(normalized_laplacian))
    normalized_laplacian ./= scale

    # √M is the exact null vector of the normalized Laplacian (the constant mode), so it overlaps the
    # lowest harmonics and seeds the iteration deterministically.
    seed = sqrt.(M)
    F = factorize_laplacian(normalized_laplacian, σ, mesh)

    krylovdim = N_modes + 12
    _, ϕ, _ = eigsolve(x -> F \ x, seed, N_modes, :LM; krylovdim, maxiter, issymmetric = true)
    while length(ϕ) < N_modes # Rerun at higher Krylov dimension if too few harmonics were generated
        krylovdim += 12
        maxiter += 100
        _, ϕ, _ = eigsolve(x -> F \ x, seed, N_modes, :LM; krylovdim, maxiter, issymmetric = true)
    end

    # M^{-1/2}·x maps each eigenvector of M^{-1/2} L M^{-1/2} back to a generalized eigenvector of (L, M).
    harmonics = [inv_sqrt_mass * x for x ∈ ϕ[1:N_modes]]

    return reduce(hcat, harmonics)
end

###
### Isoenergy Quadrature Rule
###

# Relative moment residual above which a compressed rule is reported as under-resolved. A rule whose
# non-negativity constraint is slack reproduces its moment target to roundoff, so healthy and clipped
# sheets sit around `1e-16` and `1e-2` respectively and the cutoff is placed between them.
const MOMENT_RESIDUAL_TOLERANCE = 1e-8

"""
    surface_moments(loc::SurfaceLocator{D}, ϕ::AbstractMatrix, integration_mesh::SimplexMesh{D}, v::Function) -> (moments, dropped_mass)

Fine-mesh moments of the coarse harmonic basis `ϕ` against the isoenergy measure `dμ = dS / |v|`: for each simplex of the fine `integration_mesh`, `ϕ` is interpolated at its centroid through the coarse mesh indexed by `loc`, weighted by that simplex's measure `∫ dS / |v|`, and accumulated. Fine points with no faithful coarse counterpart (an IBZ mask seam) are skipped; their `dropped_mass` is returned as a consistency diagnostic.
"""
function surface_moments(loc::SurfaceLocator{D}, ϕ::AbstractMatrix, integration_mesh::SimplexMesh{D}, v::Function) where {D}
    n_basis = size(ϕ, 2)
    moments = zeros(n_basis)
    dropped = 0.0
    for i ∈ eachindex(integration_mesh.simplices)
        measure = integrate_over_simplex(k -> 1 / norm(v(k)), i, integration_mesh)
        simplex = integration_mesh.simplices[i]
        centroid = sum(integration_mesh.vertices[k] for k ∈ simplex) / D
        ϕ_centroid = interpolate(loc, ϕ, centroid)
        if isnan(ϕ_centroid[1])
            dropped += measure
            continue
        end
        @. moments += measure * ϕ_centroid
    end
    return moments, dropped
end

"""
    fine_informed_mass(loc::SurfaceLocator{D}, integration_mesh::SimplexMesh{D}, ρ::Function) -> Vector

Vertex masses of the coarse sampling mesh indexed by `loc`, attributed from the fine `integration_mesh`: each fine simplex's mass `ρ(centroid)·area` is scattered onto the coarse vertices it maps to. More faithful than the coarse mesh's own lumping, and mass-conserving up to fine points dropped at an IBZ mask seam.
"""
function fine_informed_mass(loc::SurfaceLocator{D}, integration_mesh::SimplexMesh{D}, ρ::Function) where {D}
    fine_centroids = centroids(integration_mesh)
    masses = [ρ(fine_centroids[i]) * volume(i, integration_mesh) for i ∈ eachindex(integration_mesh.simplices)]
    return scatter_to_vertices(loc, fine_centroids, masses)
end

# Values of the vertex-sampled basis `ϕ` at the simplex centroids of `mesh`: the linear interpolant
# through a simplex's D vertices, evaluated at their barycenter.
function centroid_values(mesh::SimplexMesh{D}, ϕ::AbstractMatrix) where {D}
    basis_values = Matrix{Float64}(undef, length(mesh.simplices), size(ϕ, 2))
    for (i, simplex) ∈ enumerate(mesh.simplices), j ∈ axes(ϕ, 2)
        basis_values[i,j] = sum(k -> ϕ[k,j], simplex) / D
    end
    return basis_values
end

# Distance from each candidate in `nodes` to the `ibz` boundary, and the skin depth `δ` within which a
# candidate is barred from selection. `δ` is half the spacing of `Ns` retained nodes: in 2D they share the
# mesh's whole contour with the `Ns + 1` gaps between them; in 3D each claims an area `4δ²` of the surface
# left once the seam band of width `δ` along the boundary is removed, so
# `boundary_perimeter·δ + 4Ns·δ² = surface_measure`.
function skin_depth(mesh::SimplexMesh{D}, nodes, ibz::ConvexPolytope{D}, Ns::Int) where {D}
    boundary_distance = [distance_to_boundary(x, ibz) for x ∈ nodes]
    surface_measure = sum(volume(i, mesh) for i ∈ eachindex(simplices(mesh)))
    if D == 2
        δ = 0.5 * surface_measure / (Ns + 1)
    else
        boundary_perimeter = perimeter(mesh, ibz)
        δ = (boundary_perimeter/8Ns) *
            (sqrt(1 + 16 * surface_measure * Ns / boundary_perimeter^2) - 1)
    end
    return boundary_distance, δ
end

"""
    isoenergy_quadrature_rule(mesh::SimplexMesh, v::Function, ϕ::AbstractMatrix; moments, ibz, reject_boundary, warn_residual) -> (nodes, weights, residual)

Compressed quadrature rule for the isoenergy integral `∫ g / |v| dS`, drawing its nodes from the centroids of `mesh` and using the basis `ϕ` sampled at the mesh vertices (one column per basis function). By default the rule reproduces the moments of `ϕ` against the measure of `mesh` itself. Supplying `moments` — the same basis integrated on a finer mesh, as [`surface_moments`](@ref) returns — makes the weights track that finer measure while the candidate nodes and the harmonic basis stay on the coarse `mesh`, keeping the harmonic eigensolve tractable and the node positions stable across refinement. Also returns the moment `residual` relative to the norm of the moment target, a conditioning diagnostic warned on by [`maxvol_quadrature`](@ref) unless `warn_residual = false`. With `reject_boundary = true`, centroids within a skin depth of the `ibz` boundary are barred from selection, so their symmetry images stay distinct under BZ unfolding.
"""
function isoenergy_quadrature_rule(mesh::SimplexMesh{D}, v::Function, ϕ::AbstractMatrix;
                                   moments::Union{Nothing,AbstractVector} = nothing,
                                   ibz::Union{Nothing,ConvexPolytope{D}} = nothing,
                                   reject_boundary::Bool = true,
                                   warn_residual::Bool = true) where {D}
    Ns = size(ϕ, 2)
    moments === nothing || length(moments) == Ns ||
        throw(DimensionMismatch("moments has $(length(moments)) entries but ϕ has $Ns basis functions"))

    nodes = centroids(mesh)
    basis_values = centroid_values(mesh, ϕ)

    # Masses of the isoenergy measure ∫ · / |v| dS at the centroids, which set the selection.
    measure = [integrate_over_simplex(k -> 1 / norm(v(k)), i, mesh) for i ∈ eachindex(nodes)]
    moment_target = moments === nothing ? basis_values' * measure : moments

    boundary_distance, δ = nothing, 0.0
    if reject_boundary
        ibz === nothing && throw(ArgumentError("reject_boundary = true requires the ibz polytope"))
        boundary_distance, δ = skin_depth(mesh, nodes, ibz, Ns)
    end

    indices, weights, residual =
        maxvol_quadrature(basis_values, measure; boundary_distance, reject_boundary, δ,
                          moments = moment_target, warn_residual)

    return nodes[indices], weights, residual
end

# Distance from `x` to the nearest face of `polytope`. Each face is `{y : normal·y ≤ offset}` with unit
# outward normal, so `offset − normal·x` is the signed distance (positive inside); the minimum over faces
# is the distance to the boundary of a point interior to the polytope.
function distance_to_boundary(x::SVector{D,Float64}, polytope::ConvexPolytope{D}) where {D}
    return minimum(face.offset - dot(face.normal, x) for face ∈ polytope.faces)
end

"""
    maxvol_quadrature(basis_values::AbstractMatrix, measure::AbstractVector; boundary_distance, reject_boundary, δ, moments, warn_residual) -> (indices, weights, residual)

Quadrature rule for the discrete `measure` over points with generator values `basis_values[i,k] = φₖ(xᵢ)`: the measure-weighted Fekete point `indices`, their positive `weights`, and the moment `residual` relative to the norm of the moment target. The retained nodes reproduce the `moments`, which default to `basis_valuesᵀ·measure`, the discrete moments of the measure; a caller may pass `moments` computed on a finer mesh so the weights track that mesh's measure. With `reject_boundary = true`, candidates within the skin depth `δ` of the nearest IBZ face (measured by `boundary_distance`) are barred from selection, keeping selected nodes off the mirror planes so their images stay distinct under BZ unfolding.

Warns when a full-length selection misses its moment target by more than `MOMENT_RESIDUAL_TOLERANCE`, which is the non-negativity constraint binding and driving weights to zero. A selection shorter than the basis is left to the eligibility warning instead. Set `warn_residual = false` to stay silent and inspect the returned `residual`, as a caller sweeping many surfaces does. A rank-deficient selection would also carry a residual without any weight having been dropped; nothing here distinguishes that case.
"""
function maxvol_quadrature(basis_values::AbstractMatrix, measure::AbstractVector;
                           boundary_distance::Union{Nothing,AbstractVector{Float64}} = nothing,
                           reject_boundary::Bool = true,
                           δ = 0.0,
                           moments::AbstractVector = basis_values' * measure,
                           warn_residual::Bool = true)
    n_candidates = length(measure)
    size(basis_values, 1) == n_candidates ||
        throw(DimensionMismatch("basis_values has $(size(basis_values, 1)) rows but measure has $n_candidates entries"))
    n_basis = size(basis_values, 2)

    # Weighting by √measure makes the maxvol selection reproduce the weighted Fekete points.
    # n_candidates × n_basis, weighted_basis[i,k] = √measureᵢ · φₖ(xᵢ)
    weighted_basis = sqrt.(measure) .* basis_values

    # Skin-depth rejection bars candidates within δ of the IBZ boundary; otherwise every row is selectable.
    selectable = trues(n_candidates)
    if reject_boundary
        boundary_distance === nothing &&
            throw(ArgumentError("reject_boundary = true requires boundary_distance"))
        @inbounds for i ∈ eachindex(selectable)
            boundary_distance[i] < δ && (selectable[i] = false)
        end
    end

    # Column-pivoted QR of the transposed weighted basis pivots on its rows; the leading pivots are the
    # maxvol node set, drawn only from the selectable rows.
    selectable_indices = findall(selectable)
    length(selectable_indices) < n_basis &&
        @warn "only $(length(selectable_indices)) eligible candidates for $n_basis requested nodes; the quadrature rule will be under-resolved"
    n_selected = min(n_basis, length(selectable_indices))
    selected = selectable_indices[qr(weighted_basis[selectable_indices, :]', ColumnNorm()).p[1:n_selected]]

    # The selected nodes alone reproduce the moments.
    weights = vec(nonneg_lsq(Matrix(basis_values[selected, :]'), moments; alg = :nnls))
    residual = norm(basis_values[selected, :]' * weights - moments) / norm(moments)

    kept = findall(!iszero, weights)

    # A full-length selection interpolates the moments exactly while the non-negativity constraint is
    # slack, so a residual here is that constraint binding: the rule no longer resolves its own basis.
    warn_residual && n_selected == n_basis && residual > MOMENT_RESIDUAL_TOLERANCE &&
        @warn "compressed rule under-resolves its basis; the non-negativity constraint is binding" relative_residual = residual dropped = n_selected - length(kept) nodes = n_selected

    return selected[kept], weights[kept], residual
end

"""
    ibz_quadrature_rule(ε::Function, v::Function, crystal::Crystal, energy_window, N_s::Int, N_ε::Int; initial_depth, refinement_depth, sampling_depth, progress, reject_boundary) -> (nodes, weights, residuals)

Quadrature rule over the IBZ for the symmetric `energy_window` centered at the Fermi energy, in bare `d^Dk` measure. The marching seed grid holds `2^initial_depth + 1` points per axis, refined by `refinement_depth` further adaptive levels. In 3D each sheet's harmonics and candidate nodes come from a coarse `sampling_mesh` built at the fixed `sampling_depth` with no adaptive escalation (its resolution sets the harmonic-eigensolve cost and candidate-node spacing, independent of `refinement_depth`), while the measure is integrated on the fine mesh, with the `N_s` fine moments compressed by the same maxvol rule to order `N_s` nodes; 2D uses that rule on a single mesh and ignores `sampling_depth`. With `reject_boundary = true`, nodes are kept a skin depth clear of the IBZ boundary so their images stay distinct under unfolding. Also returns the per-sheet moment `residuals`, relative to each sheet's moment target, and warns once naming any energy levels whose residual exceeds `MOMENT_RESIDUAL_TOLERANCE`. Set `progress = true` for a progress bar over the energy levels.
"""
function ibz_quadrature_rule(ε::Function, v::Function, crystal::Crystal, energy_window, N_s::Int, N_ε::Int; initial_depth::Int, refinement_depth::Int, sampling_depth::Int = 8, progress::Bool = false, reject_boundary::Bool = true)
    energy_nodes, energy_weights = gausslegendre(N_ε)
    energy_levels = energy_window * energy_nodes
    energy_weights = energy_window * energy_weights

    ibz = irreducible_brillouin_zone(crystal)
    bounds = axis_aligned_bounding_box(ibz)

    # One (min,max) extent per axis, so the axis count is the spatial dimension.
    D = length(bounds)

    # The fine measure is scattered onto the coarse sampling mesh, so the coarse mesh must not be finer than
    # the integration mesh — otherwise coarse vertices receive no mass and the harmonic eigensolve breaks.
    D == 2 || sampling_depth ≤ refinement_depth ||
        throw(ArgumentError("sampling_depth ($sampling_depth) must be ≤ refinement_depth ($refinement_depth)"))

    nodes = Vector{SVector{D,Float64}}(undef, 0)
    weights = Vector{Float64}(undef, 0)
    residuals = Vector{Float64}(undef, 0)
    # Energy levels whose sheet came back under-resolved, reported once at the end so the sweep does
    # not warn per sheet and the offending levels stay identifiable.
    under_resolved = Float64[]
    ρ(k) = 1 / norm(v(k))

    p = Progress(length(energy_levels); desc = "Generating isoenergy surface quadrature rules...", enabled = progress)
    for i ∈ eachindex(energy_levels)
        integration_mesh = level_set(ε, energy_levels[i], bounds..., initial_depth;
            refinement_depth,
            mask = ibz,
            weld = true
        )

        # Skip empty energy levels, and levels whose sheet welded down to no simplices at all:
        # `merge_vertices` keeps every merged vertex while dropping the degenerate simplices, so a
        # grazing level set leaves a vertex bag with no surface for `skin_depth` to measure.
        (length(vertices(integration_mesh)) == 0 || isempty(simplices(integration_mesh))) &&
            (next!(p); continue)

        if D == 2
            # 2D contours are cheap to build and their compressed weights stay positive, so a single mesh
            # with the maxvol rule is retained.
            ϕ = mesh_harmonics(integration_mesh, ρ, N_s)
            surface_nodes, surface_weights, surface_residual =
                isoenergy_quadrature_rule(integration_mesh, v, ϕ; ibz, reject_boundary,
                                          warn_residual = false)
        else
            # 3D: harmonics and candidate nodes on a fixed coarse sampling mesh (keeping the eigensolve
            # tractable and node positions stable across refinement), the measure on the fine mesh.
            coarse = sampling_mesh(ε, energy_levels[i], bounds..., initial_depth; sampling_depth, mask = ibz)
            (length(vertices(coarse)) == 0 || isempty(simplices(coarse))) && (next!(p); continue)
            loc = surface_locator(coarse)
            M = fine_informed_mass(loc, integration_mesh, ρ)
            ϕ = mesh_harmonics(coarse, M, N_s)
            size(ϕ, 2) < N_s &&
                @warn "coarse sampling mesh has too few vertices for $N_s harmonics; raise sampling_depth" level = energy_levels[i] harmonics = size(ϕ, 2)
            moments, _ = surface_moments(loc, ϕ, integration_mesh, v)
            surface_nodes, surface_weights, surface_residual =
                isoenergy_quadrature_rule(coarse, v, ϕ; moments, ibz, reject_boundary,
                                          warn_residual = false)
        end

        append!(nodes, surface_nodes)
        append!(weights, surface_weights * energy_weights[i])
        push!(residuals, surface_residual)
        surface_residual > MOMENT_RESIDUAL_TOLERANCE && push!(under_resolved, energy_levels[i])

        next!(p)
    end

    isempty(under_resolved) ||
        @warn "$(length(under_resolved)) of $(length(residuals)) surfaces under-resolve the harmonic basis; raise refinement_depth or lower N_s" levels = under_resolved max_residual = maximum(residuals)

    return nodes, weights, residuals
end

"""
    bz_quadrature_rule(ε::Function, v::Function, crystal::Crystal, energy_window, N_s::Int, N_ε::Int; initial_depth, refinement_depth, sampling_depth, progress, reject_boundary) -> (nodes, weights, residuals)

Quadrature rule over the full BZ for the symmetric `energy_window` centered at the Fermi energy, unfolding the IBZ rule by the point group; weights carry the `d^Dk/(2π)^D` measure. The marching seed grid holds `2^initial_depth + 1` points per axis, refined by `refinement_depth` further adaptive levels. In 3D `sampling_depth` sets the fixed coarse sampling-mesh resolution (harmonic eigensolve and candidate nodes), independent of `refinement_depth`; 2D ignores it. With `reject_boundary = true`, IBZ nodes stay a skin depth clear of the zone boundary so their images remain distinct under unfolding and no post-merge is needed. Also returns the per-sheet relative moment `residuals` from the IBZ rule. Set `progress = true` for a progress bar over the energy levels.
"""
function bz_quadrature_rule(ε::Function, v::Function, crystal::Crystal, energy_window, N_s::Int, N_ε::Int; initial_depth::Int, refinement_depth::Int, sampling_depth::Int = 8, progress::Bool = false, reject_boundary::Bool = true)
    ibz_nodes, ibz_weights, residuals = ibz_quadrature_rule(ε, v, crystal, energy_window, N_s, N_ε; initial_depth, refinement_depth, sampling_depth, progress, reject_boundary)

    symmetry_group = point_group(crystal)
    group_order = order(symmetry_group)
    N_ibz = length(ibz_nodes)
    N = group_order * N_ibz
    nodes = typeof(ibz_nodes)(undef, N)
    weights = zeros(Float64, N)

    for (i, operation) ∈ enumerate(operations(symmetry_group))
        start = (i-1) * N_ibz + 1
        stop  = i * N_ibz
        nodes[start:stop] = map(x -> operation * x, ibz_nodes)
        weights[start:stop] = ibz_weights
    end

    # Normalize the full BZ integral by (2π)^D
    D = length(eltype(nodes))
    weights ./= (2π)^D

    return nodes, weights, residuals
end

