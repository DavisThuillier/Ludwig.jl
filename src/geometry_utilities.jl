"""
    in_polygon(k, p, atol=1e-12)

Test whether the 2D point `k` lies inside the polygon `p`, including its vertices and edges.
"""
function in_polygon(k, p, atol = 1e-12)
    is_vertex = false
    for vertex in p
        isapprox(k, vertex, atol = atol) && (is_vertex = true)
    end
    is_vertex && return true

    w = winding_number(p, k, atol) # Assumes p is counterclockwise ordered
    return !(abs(w) < atol) # Returns false when w ≈ 0
end

"""
    winding_number(v, center, atol=1e-12)

Return the winding number of the closed polygon with vertices `v` about the point `center`.
"""
function winding_number(v, center, atol = 1e-12)
    w = 0.0
    for i in eachindex(v)
        j = (i == length(v)) ? 1 : i + 1

        # Vertices i and j relative to center
        xi, yi = v[i][1] - center[1], v[i][2] - center[2]
        xj, yj = v[j][1] - center[1], v[j][2] - center[2]

        if yi * yj < 0 # [vᵢvⱼ] crosses the x-axis
            m = (xj - xi) / (yi - yj)
            r = xi + m * yi # x-coordinate of intersection of [vᵢvⱼ] with x-axis
            if abs(r) < atol
                m > 0 ? (w += 0.5) : (w -= 0.5)
            elseif r > 0
                yi < 0 ? (w += 1) : (w -= 1)
            end
        elseif abs(yi) < atol && abs(yj) > atol && xi > 0 # vᵢ on the positive x-axis
            yj > 0 ? (w += 0.5) : (w -= 0.5)
        elseif abs(yi) > atol && abs(yj) < atol && xj > 0 # vⱼ on the positive x-axis
            yi < 0 ? (w += 0.5) : (w -= 0.5)
        end
    end
    return w
end

"""
    diameter(p)

Return the maximal distance between two points in `p`.
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
    boundary_distance(p, polygon)

Return the distance from the 2D point `p` to the nearest edge of the closed `polygon`.
"""
function boundary_distance(p, polygon)
    n = length(polygon)
    return minimum(1:n) do k
        a = polygon[k]
        b = polygon[mod1(k + 1, n)]
        t = clamp(dot(p - a, b - a) / dot(b - a, b - a), 0.0, 1.0)
        norm(p - (a + t * (b - a)))
    end
end

"""
    signed_area(x, y, z)

Return the signed area of the 2D triangle with vertices `x`, `y`, `z`.
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
"""
intersection(a1, v1, a2, v2) = param_intersection(a1, v1, a2, v2)[1]

"""
    perpendicular_bisector_intersection(v1, v2)

Return the point where the perpendicular bisectors of the segments from the origin to `v1`
and from the origin to `v2` meet.
"""
function perpendicular_bisector_intersection(v1, v2)
    return intersection(0.5 * v1, [v1[2], -v1[1]], 0.5 * v2, [v2[2], -v2[1]])
end

"""
    poly_area(poly, c)

Return the area of the 2D polygon with vertex set `poly`, oriented about the interior
point `c`.
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

Return an approximation to a root of monotone `f` in the bracket `[a, b]` by bisection.
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

Return the index of the edge of polygon `region` on which point `p` lies, or `0`.
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
