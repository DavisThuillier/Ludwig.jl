"""
    in_polygon(k, p, atol=1e-12)

Test whether the 2D point `k` lies inside the polygon `p`, including its vertices.

`p` is a sequence of vertices (e.g. a `Vector{<:AbstractVector}`); the polygon is implicitly
closed by joining the last vertex to the first. Behavior for points lying on the edges
between adjacent vertices is not guaranteed to be stable.

# Implementation
A point coincident with a vertex (within `atol`) returns `true` directly; otherwise the
result is determined by the winding number of `p` about `k`.

# Examples
```jldoctest
julia> square = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];

julia> in_polygon([0.5, 0.5], square)
true

julia> in_polygon([2.0, 0.5], square)
false
```
"""
function in_polygon(k, p, atol = 1e-12)
    is_vertex = false
    for vertex in p
        isapprox(k, vertex, atol = atol) && (is_vertex = true)
    end
    is_vertex && return true

    w = winding_number(p .- Ref(k), atol) # Winding number of polygon wrt k
    return !(abs(w) < atol) # Returns false when w ≈ 0
end

"""
    winding_number(v, atol=1e-12)

Return the winding number of the closed polygon with vertices `v` about the origin.

`v` is a sequence of 2D points; the polygon is implicitly closed by joining the last vertex
to the first. The result is an integer for generic inputs and a half-integer when the
origin lies on an edge or vertex (within `atol`).

# Examples
```jldoctest
julia> winding_number([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
1.0

julia> winding_number([[2.0, 1.0], [1.0, 2.0], [0.0, 1.0], [1.0, 0.0]])
0.0
```
"""
function winding_number(v, atol = 1e-12)
    w = 0
    for i in eachindex(v)
        j = (i == length(v)) ? 1 : i + 1

        if v[i][2] * v[j][2] < 0 # [vᵢvⱼ] crosses the x-axis
            m = (v[j][1] - v[i][1]) / (v[i][2] - v[j][2])
            r = v[i][1] + m * v[i][2]  # x-coordinate of intersection of [vᵢvⱼ] with x-axis
            if abs(r) < atol
                m > 0 ? (w += 0.5) : (w -= 0.5)
            elseif r > 0
                v[i][2] < 0 ? (w += 1) : (w -= 1)
            end
        elseif abs(v[i][2]) < atol && abs(v[j][2]) > atol && v[i][1] > 0 # vᵢ on the positive x-axis
            v[j][2] > 0 ? (w += 0.5) : (w -= 0.5)
        elseif abs(v[i][2]) > atol && abs(v[j][2]) < atol && v[j][1] > 0 # vⱼ on the positive x-axis
            v[i][2] < 0 ? (w += 0.5) : (w -= 0.5)
        end
    end
    return w
end

"""
    diameter(p)

Return the maximal distance between two points in `p`.

# Examples
```jldoctest
julia> diameter([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
1.4142135623730951
```
"""
function diameter(p)
    diameter = 0.0
    for i in eachindex(p)
        for j in i:length(p)
            d = norm(p[i]-p[j])
            d > diameter && (diameter = d)
        end
    end

    return diameter
end

"""
    signed_area(x, y, z)

Return the signed area of the 2D triangle with vertices `x`, `y`, `z`.

The result is positive when ``x \\to y \\to z`` traces the triangle counter-clockwise and
negative when clockwise.

# Examples
```jldoctest
julia> signed_area([0.0, 0.0], [1.0, 0.0], [0.0, 1.0])
0.5

julia> signed_area([0.0, 0.0], [0.0, 1.0], [1.0, 0.0])
-0.5
```
"""
function signed_area(x, y, z)
    return 0.5 * (-y[1] * x[2] + z[1] * x[2] + x[1] * y[2] - z[1] * y[2] - x[1]*z[2] + y[1] * z[2])
end

"""
    param_intersection(a1, v1, a2, v2; rtol=1e-12)

Return the intersection point of the parametric lines ``\\mathbf{a}_1 + t_1 \\mathbf{v}_1``
and ``\\mathbf{a}_2 + t_2 \\mathbf{v}_2``, together with the parameter pair ``(t_1, t_2)``
at the intersection.

The lines are treated as parallel — and `([NaN, NaN], [NaN, NaN])` is returned — when
``|\\det[v_1\\,{-v_2}]| < \\mathrm{rtol}\\,\\|v_1\\|\\,\\|v_2\\|``. Since
``\\det[v_1\\,{-v_2}] = \\|v_1\\|\\,\\|v_2\\|\\,\\sin\\theta``, this is equivalent to a
threshold on ``|\\sin\\theta|``: the default `rtol = 1e-12` flags lines closer than
``\\sim 10^{-12}`` rad to parallel.

See also `intersection`.

# Examples
```jldoctest
julia> p, t = param_intersection([0.0, 0.0], [1.0, 0.0], [1.0, -1.0], [0.0, 1.0]);

julia> p
2-element Vector{Float64}:
 1.0
 0.0

julia> t
2-element Vector{Float64}:
 1.0
 1.0
```
"""
function param_intersection(a1, v1, a2, v2; rtol = 1e-12)
    V = hcat(v1, -v2)
    abs(det(V)) < rtol * norm(v1) * norm(v2) && return [NaN, NaN], [NaN, NaN]

    t = inv(V) * (a2 - a1)
    return a1 + t[1] * v1, t
end

"""
    intersection(a1, v1, a2, v2)

Return the intersection point of the parametric lines ``\\mathbf{a}_1 + t_1 \\mathbf{v}_1``
and ``\\mathbf{a}_2 + t_2 \\mathbf{v}_2``, or `[NaN, NaN]` if the lines are parallel.

Equivalent to `param_intersection(a1, v1, a2, v2)[1]`; use `param_intersection` when
the parameter values are also needed.

# Examples
```jldoctest
julia> intersection([0.0, 0.0], [1.0, 0.0], [1.0, -1.0], [0.0, 1.0])
2-element Vector{Float64}:
 1.0
 0.0
```
"""
intersection(a1, v1, a2, v2) = param_intersection(a1, v1, a2, v2)[1]

"""
    perpendicular_bisector_intersection(v1, v2)

Return the point where the perpendicular bisectors of the segments from the origin to `v1`
and from the origin to `v2` meet.

This is the circumcenter of the triangle with vertices `[0, 0]`, `v1`, and `v2`, i.e. the
point equidistant from all three. Returns `[NaN, NaN]` when the origin, `v1`, and `v2` are
colinear.

# Examples
```jldoctest
julia> perpendicular_bisector_intersection([2.0, 0.0], [0.0, 2.0])
2-element Vector{Float64}:
 1.0
 1.0
```
"""
function perpendicular_bisector_intersection(v1, v2)
    return intersection(0.5 * v1, [v1[2], -v1[1]], 0.5 * v2, [v2[2], -v2[1]])
end

"""
    poly_area(poly, c)

Return the area of the 2D polygon with vertex set `poly`, oriented about the interior
point `c`.

`c` must lie inside the convex hull of `poly`; otherwise the result is meaningless. The
returned value is positive for non-degenerate inputs.

# Implementation
Vertices are sorted by polar angle about `c` and the shoelace formula is applied to the
resulting traversal. When `c` lies outside the convex hull, this sort can produce a
self-intersecting polygon.

# Examples
```jldoctest
julia> poly_area([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], [0.5, 0.5])
1.0
```
"""
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

###
### Root finding
###

"""
    bisect(f, a, b; iter=64, atol=eps(Float64))

Return an approximation to a root of `f` in the bracket `[a, b]` by bisection.

`f(a)` and `f(b)` must have opposite signs. The loop terminates either when `iter`
bisection steps have run or when the bracket width drops below `atol`, whichever
comes first; the midpoint of the final bracket is returned. With `atol = 0` the
function always runs the full `iter` halvings.

# Examples
```jldoctest
julia> bisect(x -> x^2 - 2, 1.0, 2.0)
1.414213562373095
```
"""
function bisect(f, a, b; iter = 64, atol = eps(Float64))
    fa = f(a)
    for _ in 1:iter
        mid = (a + b) / 2
        fmid = f(mid)
        fmid * fa > 0 ? (a = mid; fa = fmid) : (b = mid)
        abs(b - a) < atol && break
    end
    return (a + b) / 2
end

"""
    edge_index(p, region[; tol]) -> Int

Return the 1-based index of the edge of polygon `region` on which point `p` lies, or `0`.

Edge `k` is the segment from `region[k]` to `region[mod1(k+1, n)]`. Point `p` is considered
to lie on that edge when it is collinear with the segment (cross-product residual per unit
length ≤ `tol`) and the projection parameter `t ∈ [0, 1]` (within `tol`).

# Examples
```jldoctest
julia> square = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];

julia> edge_index([0.5, 0.0], square)
1

julia> edge_index([0.5, 0.5], square)
0
```
"""
function edge_index(p, region; tol = 1e-8)
    n = length(region)
    for k in 1:n
        a  = region[k]
        b  = region[mod1(k + 1, n)]
        v  = b - a
        w  = p - a
        lv = norm(v)
        lv < eps() && continue
        abs(v[1] * w[2] - v[2] * w[1]) / lv > tol && continue
        t = dot(w, v) / dot(v, v)
        -tol ≤ t ≤ 1 + tol && return k
    end
    return 0
end
