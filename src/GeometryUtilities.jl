module GeometryUtilities

import LinearAlgebra: det, norm
export in_polygon, diameter, signed_area, intersection, param_intersection, perpendicular_bisector_intersection, poly_area

function in_polygon(k, p, atol = 1e-12)
    is_vertex = false
    for vertex in p
        isapprox(k, vertex, atol = atol) && (is_vertex = true)
    end
    is_vertex && return true

    w = winding_number(p .- Ref(k), atol) # Winding number of polygon wrt k
    return !(abs(w) < atol) # Returns false is w ≈ 0
end

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

function signed_area(x, y, z)
    return 0.5 * (-y[1] * x[2] + z[1] * x[2] + x[1] * y[2] - z[1] * y[2] - x[1]*z[2] + y[1] * z[2])
end

function param_intersection(a1, v1, a2, v2)
    V = hcat(v1, -v2)
    det(V) == 0 && return [NaN, NaN] # No intersection

    t = inv(V) * (a2 - a1)
    return a1 + t[1] * v1, t
end

intersection(a1, v1, a2, v2) = param_intersection(a1, v1, a2, v2)[1]

function perpendicular_bisector_intersection(v1, v2)
    return intersection(0.5 * v1, [v1[2], -v1[1]], 0.5 * v2, [v2[2], -v2[1]])
end

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

end