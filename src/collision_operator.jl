
"""
    fermi_dirac(E, T)

Fermi-Dirac occupation for energy `E` and temperature `T` (both in energy units).
"""
fermi_dirac(E, T) = 1 / (exp(E/T) + 1)

###
### Electron-electron collision integral
###

# Fold a fractional momentum into the marching cell `[-1/2, 1/2)^D` by minimal image (subtracting the
# nearest reciprocal-lattice vector), matching the Γ-centered seed grid. For a lattice-periodic band this
# agrees with a `[0, 1)^D` fold; for a Γ-centered continuum pocket it keeps the whole pocket in the cell.
minimal_image(x) = x .- round.(x)

"""
    electron_electron(ki, kj, lattice, ε, v, T, Weff_squared; initial_depth, refinement_depth, kwargs...)

Electron-electron collision integral `Lij` for Cartesian momenta `ki`, `kj`, with `ε`, `v`, and
`Weff_squared` evaluated in the Cartesian basis. The δ-surface integrals are marched on a
`2^initial_depth + 1` seed grid per axis and refined to a total subdivision depth of `refinement_depth`
(which must be `≥ initial_depth`).

A non-lattice-periodic (continuum) `ε` is captured correctly only when its pocket is centered at Γ; a
pocket centered elsewhere straddles the boundary of the Γ-centered marching cell and needs the box
centered on the pocket instead.
"""
function electron_electron(ki::SVector{D,Float64}, kj::SVector{D,Float64}, wj::Float64,
    lattice::Lattice{D}, ε::Function, v::Function, T, Weff_squared::Function;
    initial_depth::Int, refinement_depth::Int, threaded = false, kwargs...
    ) where {D}

    # Basis transforms read directly off the lattice.
    rlv     = reciprocal_lattice_vectors(lattice)   # fractional -> Cartesian
    inv_rlv = transpose(primitives(lattice)) / (2π) # Cartesian -> fractional
    absdet_rlv = abs(det(rlv))                      # fractional -> Cartesian volume factor

    # Pointwise fractional -> Cartesian surface Jacobian for the δ-surface integrals
    surface_jacobian(vA, vB) = absdet_rlv / norm(transpose(rlv) * (vA - vB))

    # Energies at the Cartesian incoming momenta
    εi  = ε(ki); fdi = fermi_dirac(εi, T)
    εj  = ε(kj); fdj = fermi_dirac(εj, T)

    # Momentum sums and differences in the fractional basis (for Umklapp wrapping)
    ki_f = inv_rlv * ki
    kj_f = inv_rlv * kj
    kij_f = ki_f + kj_f
    qij_f = ki_f - kj_f

    Lij = 0.0

    ###
    ### Sum surface: ε(k3) + ε(k4) = εi + εj, with k4 = kij - k3
    ###
    sum_sampler(k3_f) = begin
        k3_c = rlv * k3_f
        k4_c = rlv * minimal_image(kij_f - k3_f)   # Umklapp handled in fractional space before mapping to Cartesian
        ε3 = ε(k3_c)
        ε4 = ε(k4_c)
        VertexSample(k3_f, ε3, ε4, ε3 + ε4 - (εi + εj))
    end
    function sum_integrand(k3_f, ε3, ε4)
        k3_c = rlv * k3_f
        k4_c = rlv * minimal_image(kij_f - k3_f)
        v3 = v(k3_c); v4 = v(k4_c)
        fd3 = fermi_dirac(ε3, T); fd4 = fermi_dirac(ε4, T)
        W_ij34 = Weff_squared(ki, kj, k3_c, k4_c; kwargs...)
        return (1 - fd3) * (1 - fd4) * W_ij34 * surface_jacobian(v3, v4)
    end
    Lij += fdj / (1 - fdi) *
           integrate_level_set(sum_sampler, sum_integrand, initial_depth, Val(D); refinement_depth, threaded)

    ###
    ### Difference surface: εi + ε(k2) = εj + ε(k4), with k4 = qij + k2
    ###
    difference_sampler(k2_f) = begin
        k2_c = rlv * k2_f
        k4_c = rlv * minimal_image(qij_f + k2_f)   # Umklapp handled in fractional space before mapping to Cartesian
        ε2 = ε(k2_c)
        ε4 = ε(k4_c)
        VertexSample(k2_f, ε2, ε4, ε2 - ε4 - (εj - εi))
    end
    function difference_integrand(k2_f, ε2, ε4)
        k2_c = rlv * k2_f
        k4_c = rlv * minimal_image(qij_f + k2_f)
        v2 = v(k2_c); v4 = v(k4_c)
        fd2 = fermi_dirac(ε2, T); fd4 = fermi_dirac(ε4, T)
        W_i2j4 = Weff_squared(ki, k2_c, kj, k4_c; kwargs...)
        W_i23j = Weff_squared(ki, k2_c, k4_c, kj; kwargs...)
        return fd2 * (1 - fd4) * (W_i2j4 + W_i23j) * surface_jacobian(v2, v4)
    end
    Lij -= ((1 - fdj) / (1 - fdi)) *
           integrate_level_set(difference_sampler, difference_integrand, initial_depth, Val(D); refinement_depth, threaded)

    return wj * π * Lij / (2π)^D
end

"""
    electron_electron(ki, kj, crystal, ε, v, T, Weff_squared; kwargs...)

Convenience overload forwarding to the `Lattice` method via `lattice(crystal)`.
"""
electron_electron(ki::SVector{D,Float64}, kj::SVector{D,Float64}, wj::Float64, crystal::Crystal{D},
    ε::Function, v::Function, T, Weff_squared::Function; kwargs...) where {D} =
    electron_electron(ki, kj, wj, lattice(crystal), ε, v, T, Weff_squared; kwargs...)


###
### Level-set integrator
###
###
### Fused, allocation-free surface integrator for `electron_electron`: it accumulates each leaf
### simplex's contribution to a scalar rather than storing a mesh. The two conjugate energies are
### carried through the refinement and read by linear interpolation at each crossing, so `ε` is never
### resampled there.
###

"""
    integrate_level_set(sampler, integrand, initial_depth, Val(D); refinement_depth, threaded=false, nchunks=Threads.nthreads())

Integrate `integrand` over the level set seeded by `sampler` on the Γ-centered `(-1/2, 1/2)^D` unit cell,
never storing a mesh. The coarse seed grid has `N = 2^initial_depth + 1` points per axis; each straddling cell is then
adaptively refined until the total subdivision depth from the unit cell reaches `refinement_depth`, so the
finest cell spans `1 / 2^refinement_depth` regardless of `initial_depth`. `refinement_depth` must be
`≥ initial_depth`. `sampler(p) -> VertexSample{D}` supplies the level value and the two conjugate energies;
`integrand(q, energy_a, energy_b) -> Float64` is evaluated at each crossing using the interpolated energies.
"""
function integrate_level_set(sampler::F, integrand::I, initial_depth::Int, ::Val{D};
                             refinement_depth::Int,
                             threaded::Bool = false,
                             nchunks::Int = Threads.nthreads()) where {F,I,D}
    initial_depth ≥ 0 || throw(ArgumentError("initial_depth must be nonnegative, got initial_depth = $initial_depth"))
    refinement_depth ≥ initial_depth || throw(ArgumentError("refinement_depth (=$refinement_depth) must be ≥ initial_depth (=$initial_depth)"))
    N = 2^initial_depth + 1
    maxdepth = refinement_depth - initial_depth
    dx = 1.0 / (N - 1)
    ncells = (N - 1)^D

    # Sample the seed grid once so each cell reads shared corners by index instead of resampling them.
    grid = seed_samples(sampler, N, dx, Val(D))

    integrate_cells(sampler, grid, integrand, N, maxdepth, 1, ncells, Val(D))
end

"""
    seed_samples(sampler, N, dx, Val(D)) -> Array{VertexSample{D},D}

Sample `sampler` at every point of the `N`-per-axis grid on the Γ-centered `(-1/2, 1/2)^D` cell
(spacing `dx`), evaluating each grid point exactly once.
"""
function seed_samples(sampler::F, N::Int, dx::Float64, ::Val{2}) where {F}
    grid = Array{VertexSample{2},2}(undef, N, N)
    @inbounds for j in 1:N, i in 1:N
        grid[i, j] = sampler(SVector((i - 1) * dx - 0.5, (j - 1) * dx - 0.5))
    end
    return grid
end

function seed_samples(sampler::F, N::Int, dx::Float64, ::Val{3}) where {F}
    grid = Array{VertexSample{3},3}(undef, N, N, N)
    @inbounds for k in 1:N, j in 1:N, i in 1:N
        grid[i, j, k] = sampler(SVector((i - 1) * dx - 0.5, (j - 1) * dx - 0.5, (k - 1) * dx - 0.5))
    end
    return grid
end

"""
    VertexSample{D}

A refinement sample: the `point`, its two conjugate energies `energy_a`/`energy_b`, and the shifted
level-function value `level_value` (= the combined energies minus the target level).
"""
struct VertexSample{D}
    point::SVector{D,Float64}
    energy_a::Float64
    energy_b::Float64
    level_value::Float64
end

"""
    crossing(A, B)

Return `(point, energy_a, energy_b)` at the level-set crossing on edge `A`–`B`, linearly interpolated by
the same parameter that locates the crossing.
"""
@inline function crossing(A::VertexSample{D}, B::VertexSample{D}) where {D}
    t = A.level_value / (A.level_value - B.level_value)
    return (A.point    + t * (B.point    - A.point),
            A.energy_a + t * (B.energy_a - A.energy_a),
            A.energy_b + t * (B.energy_b - A.energy_b))
end

@inline triangle_area(p0, p1, p2) = norm(cross(p1 - p0, p2 - p0)) / 2

"""
    segment_contribution(integrand, A, B, C)

Integrate `integrand` over the contour segment crossing the leaf triangle `A,B,C` (trapezoid rule on the
two edge crossings), returning `0.0` if the triangle does not straddle the level.
"""
function segment_contribution(integrand::I, A::VertexSample{2}, B::VertexSample{2}, C::VertexSample{2})::Float64 where {I}
    a_above = A.level_value ≥ 0.0
    b_above = B.level_value ≥ 0.0
    c_above = C.level_value ≥ 0.0
    (a_above == b_above == c_above) && return 0.0   # all on one side: no crossing

    point_1 = A.point; value_1 = 0.0
    point_2 = A.point; value_2 = 0.0
    nfound = 0
    if a_above != b_above
        crossing_point, energy_a, energy_b = crossing(A, B)
        crossing_value = integrand(crossing_point, energy_a, energy_b)
        nfound == 0 ? (point_1 = crossing_point; value_1 = crossing_value) : (point_2 = crossing_point; value_2 = crossing_value); nfound += 1
    end
    if b_above != c_above
        crossing_point, energy_a, energy_b = crossing(B, C)
        crossing_value = integrand(crossing_point, energy_a, energy_b)
        nfound == 0 ? (point_1 = crossing_point; value_1 = crossing_value) : (point_2 = crossing_point; value_2 = crossing_value); nfound += 1
    end
    if c_above != a_above
        crossing_point, energy_a, energy_b = crossing(C, A)
        crossing_value = integrand(crossing_point, energy_a, energy_b)
        nfound == 0 ? (point_1 = crossing_point; value_1 = crossing_value) : (point_2 = crossing_point; value_2 = crossing_value); nfound += 1
    end

    nfound == 2 || return 0.0
    return norm(point_2 - point_1) * (value_1 + value_2) / 2
end

"""
    triangle_contribution(integrand, P1, P2, P3, P4)

Integrate `integrand` over the iso-surface (one or two triangles) crossing the leaf tetrahedron
`P1,P2,P3,P4` (centroid rule per triangle), returning `0.0` if the tetrahedron does not straddle the
level.
"""
function triangle_contribution(integrand::I,
                               P1::VertexSample{3}, P2::VertexSample{3},
                               P3::VertexSample{3}, P4::VertexSample{3})::Float64 where {I}
    is_above = (P1.level_value ≥ 0.0, P2.level_value ≥ 0.0,
                P3.level_value ≥ 0.0, P4.level_value ≥ 0.0)
    below_count = (!is_above[1]) + (!is_above[2]) + (!is_above[3]) + (!is_above[4])
    (below_count == 0 || below_count == 4) && return 0.0   # all on one side: no surface

    samples = (P1, P2, P3, P4)

    if below_count == 1 || below_count == 3
        # 1-vs-3: one corner sits alone on its side of the level, and the surface is the single
        # triangle cut from the three edges incident to it.
        lone_side = below_count == 3
        lone = is_above[1] == lone_side ? 1 :
               is_above[2] == lone_side ? 2 :
               is_above[3] == lone_side ? 3 : 4
        other_1 = 0; other_2 = 0; other_3 = 0
        for i in 1:4
            i == lone && continue
            if other_1 == 0
                other_1 = i
            elseif other_2 == 0
                other_2 = i
            else
                other_3 = i
            end
        end
        crossing_1, energy_a, energy_b = crossing(samples[lone], samples[other_1])
        value_1 = integrand(crossing_1, energy_a, energy_b)
        crossing_2, energy_a, energy_b = crossing(samples[lone], samples[other_2])
        value_2 = integrand(crossing_2, energy_a, energy_b)
        crossing_3, energy_a, energy_b = crossing(samples[lone], samples[other_3])
        value_3 = integrand(crossing_3, energy_a, energy_b)
        return triangle_area(crossing_1, crossing_2, crossing_3) * (value_1 + value_2 + value_3) / 3
    else
        # 2-vs-2: the level cuts the four edges joining the pair below it to the pair above it,
        # giving a quadrilateral that splits into 2 triangles.
        below_1 = 0; below_2 = 0; above_1 = 0; above_2 = 0
        for i in 1:4
            if is_above[i]
                above_1 == 0 ? (above_1 = i) : (above_2 = i)
            else
                below_1 == 0 ? (below_1 = i) : (below_2 = i)
            end
        end
        # The four crossings, in cyclic order around the quadrilateral.
        crossing_1, energy_a, energy_b = crossing(samples[below_1], samples[above_1])
        value_1 = integrand(crossing_1, energy_a, energy_b)
        crossing_2, energy_a, energy_b = crossing(samples[below_1], samples[above_2])
        value_2 = integrand(crossing_2, energy_a, energy_b)
        crossing_3, energy_a, energy_b = crossing(samples[below_2], samples[above_2])
        value_3 = integrand(crossing_3, energy_a, energy_b)
        crossing_4, energy_a, energy_b = crossing(samples[below_2], samples[above_1])
        value_4 = integrand(crossing_4, energy_a, energy_b)
        # Split along the crossing_1–crossing_3 diagonal.
        return triangle_area(crossing_1, crossing_2, crossing_3) * (value_1 + value_2 + value_3) / 3 +
               triangle_area(crossing_1, crossing_3, crossing_4) * (value_1 + value_3 + value_4) / 3
    end
end

"""
    integrate_refined_triangle(sampler, integrand, A, B, C, depth, maxdepth)

Quadrisect a straddling triangle (resampling at its edge midpoints) until the size floor, then integrate
its leaf contribution; return the accumulated integral.
"""
function integrate_refined_triangle(sampler::F, integrand::I,
                                    A::VertexSample{2}, B::VertexSample{2}, C::VertexSample{2},
                                    depth::Int, maxdepth::Int)::Float64 where {F,I}
    if depth < maxdepth && signchange(A.level_value, B.level_value, C.level_value)
        AB = sampler(0.5 * (A.point + B.point))
        BC = sampler(0.5 * (B.point + C.point))
        CA = sampler(0.5 * (C.point + A.point))
        child_depth = depth + 1
        return integrate_refined_triangle(sampler, integrand, A,  AB, CA, child_depth, maxdepth) +
               integrate_refined_triangle(sampler, integrand, AB, B,  BC, child_depth, maxdepth) +
               integrate_refined_triangle(sampler, integrand, CA, BC, C,  child_depth, maxdepth) +
               integrate_refined_triangle(sampler, integrand, AB, BC, CA, child_depth, maxdepth)
    else
        return segment_contribution(integrand, A, B, C)
    end
end

"""
    integrate_refined_tetrahedron(sampler, integrand, P1, P2, P3, P4, depth, maxdepth)

Octasect a straddling tetrahedron into 8 sub-tets (resampling at its 6 edge midpoints, interior tets
fanned around the shortest octahedron diagonal) until the size floor, then integrate its leaf
contribution; return the accumulated integral.
"""
function integrate_refined_tetrahedron(sampler::F, integrand::I,
                                       P1::VertexSample{3}, P2::VertexSample{3},
                                       P3::VertexSample{3}, P4::VertexSample{3},
                                       depth::Int, maxdepth::Int)::Float64 where {F,I}
    if depth < maxdepth && signchange(P1.level_value, P2.level_value, P3.level_value, P4.level_value)
        M12 = sampler(0.5*(P1.point+P2.point)); M13 = sampler(0.5*(P1.point+P3.point)); M14 = sampler(0.5*(P1.point+P4.point))
        M23 = sampler(0.5*(P2.point+P3.point)); M24 = sampler(0.5*(P2.point+P4.point)); M34 = sampler(0.5*(P3.point+P4.point))
        child_depth = depth + 1
        # 4 corner tets
        total = integrate_refined_tetrahedron(sampler, integrand, P1, M12, M13, M14, child_depth, maxdepth) +
                integrate_refined_tetrahedron(sampler, integrand, P2, M12, M23, M24, child_depth, maxdepth) +
                integrate_refined_tetrahedron(sampler, integrand, P3, M13, M23, M34, child_depth, maxdepth) +
                integrate_refined_tetrahedron(sampler, integrand, P4, M14, M24, M34, child_depth, maxdepth)
        # 4 interior tets, fanned around the SHORTEST octahedron diagonal (Bey: bounds slivers)
        diagonal_12_34 = sum(abs2, M12.point - M34.point)
        diagonal_13_24 = sum(abs2, M13.point - M24.point)
        diagonal_14_23 = sum(abs2, M14.point - M23.point)
        if diagonal_12_34 ≤ diagonal_13_24 && diagonal_12_34 ≤ diagonal_14_23
            total += integrate_refined_tetrahedron(sampler, integrand, M12, M34, M13, M23, child_depth, maxdepth) +
                     integrate_refined_tetrahedron(sampler, integrand, M12, M34, M23, M24, child_depth, maxdepth) +
                     integrate_refined_tetrahedron(sampler, integrand, M12, M34, M24, M14, child_depth, maxdepth) +
                     integrate_refined_tetrahedron(sampler, integrand, M12, M34, M14, M13, child_depth, maxdepth)
        elseif diagonal_13_24 ≤ diagonal_14_23
            total += integrate_refined_tetrahedron(sampler, integrand, M13, M24, M12, M23, child_depth, maxdepth) +
                     integrate_refined_tetrahedron(sampler, integrand, M13, M24, M23, M34, child_depth, maxdepth) +
                     integrate_refined_tetrahedron(sampler, integrand, M13, M24, M34, M14, child_depth, maxdepth) +
                     integrate_refined_tetrahedron(sampler, integrand, M13, M24, M14, M12, child_depth, maxdepth)
        else
            total += integrate_refined_tetrahedron(sampler, integrand, M14, M23, M12, M24, child_depth, maxdepth) +
                     integrate_refined_tetrahedron(sampler, integrand, M14, M23, M24, M34, child_depth, maxdepth) +
                     integrate_refined_tetrahedron(sampler, integrand, M14, M23, M34, M13, child_depth, maxdepth) +
                     integrate_refined_tetrahedron(sampler, integrand, M14, M23, M13, M12, child_depth, maxdepth)
        end
        return total
    else
        return triangle_contribution(integrand, P1, P2, P3, P4)
    end
end

# Integrate the 2 triangles of each cell over the linear cell-index block `lo:hi` (no stored mesh).
# Cell corners are read from the pre-sampled `grid`; refinement midpoints are still sampled on demand.
function integrate_cells(sampler::F, grid::Array{VertexSample{2},2}, integrand::I, N::Int, maxdepth::Int, lo::Int, hi::Int, ::Val{2})::Float64 where {F,I}
    nx = N - 1
    total = 0.0
    @inbounds for cell in lo:hi
        i = (cell - 1) % nx + 1
        j = (cell - 1) ÷ nx + 1
        S00 = grid[i,   j]
        S10 = grid[i+1, j]
        S01 = grid[i,   j+1]
        S11 = grid[i+1, j+1]
        total += integrate_refined_triangle(sampler, integrand, S00, S10, S11, 0, maxdepth)
        total += integrate_refined_triangle(sampler, integrand, S00, S11, S01, 0, maxdepth)
    end
    return total
end

# Integrate the 6 Kuhn tets of each cell over the linear cell-index block `lo:hi` (no stored mesh).
# Cell corners are read from the pre-sampled `grid`; refinement midpoints are still sampled on demand.
function integrate_cells(sampler::F, grid::Array{VertexSample{3},3}, integrand::I, N::Int, maxdepth::Int, lo::Int, hi::Int, ::Val{3})::Float64 where {F,I}
    nx = N - 1
    total = 0.0
    @inbounds for cell in lo:hi
        remaining = cell - 1
        i = remaining % nx + 1; remaining ÷= nx
        j = remaining % nx + 1
        k = remaining ÷ nx + 1
        C0 = grid[i,   j,   k]     # (0,0,0)
        C1 = grid[i+1, j,   k]     # (1,0,0)
        C2 = grid[i,   j+1, k]     # (0,1,0)
        C3 = grid[i+1, j+1, k]     # (1,1,0)
        C4 = grid[i,   j,   k+1]   # (0,0,1)
        C5 = grid[i+1, j,   k+1]   # (1,0,1)
        C6 = grid[i,   j+1, k+1]   # (0,1,1)
        C7 = grid[i+1, j+1, k+1]   # (1,1,1)
        # 6 Kuhn tets sharing the main diagonal 0–7
        total += integrate_refined_tetrahedron(sampler, integrand, C0, C1, C3, C7, 0, maxdepth)
        total += integrate_refined_tetrahedron(sampler, integrand, C0, C1, C5, C7, 0, maxdepth)
        total += integrate_refined_tetrahedron(sampler, integrand, C0, C2, C3, C7, 0, maxdepth)
        total += integrate_refined_tetrahedron(sampler, integrand, C0, C2, C6, C7, 0, maxdepth)
        total += integrate_refined_tetrahedron(sampler, integrand, C0, C4, C5, C7, 0, maxdepth)
        total += integrate_refined_tetrahedron(sampler, integrand, C0, C4, C6, C7, 0, maxdepth)
    end
    return total
end
