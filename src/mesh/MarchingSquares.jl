module MarchingSquares

import StaticArrays: SVector
import DataStructures: OrderedDict
import LinearAlgebra: norm, det, dot

export Isoline, IsolineBundle, contour_intersection

"""
Representation of a contour as an ordered set of discrete points

# Fields
- `points`: Vector of points in contour
- `isclosed`: Boolean which is `true` if the contour returned to the starting point when being generated
- `arclength`: Length of the contour
"""
struct Isoline
    points::Vector{SVector{2,Float64}}
    isclosed::Bool
    arclength::Real
end

"""
Collection of contours generated from a matrix.

# Fields
- `isolines`: Vector of Isolines
- `level`: Constant value along contours in `isolines`
"""
struct IsolineBundle
    isolines::Vector{Isoline}
    level::Float64
end

# Top, Bottom, Left, Right
T, B, R, L = 0x01, 0x02, 0x04, 0x08
crossing_lookup = [L|B, B|R, L|R, T|R, 0x0, T|B, L|T, L|T, T|B, 0x0, T|R, L|R, B|R, L|B] # Ignore the ambiguous cases of two crossings in one cell, with the assumption that chosen gridding is small enough

function in_polygon(k, p, atol = 1e-12)
    is_vertex = false
    for vertex in p
        isapprox(k, vertex, atol = atol) && (is_vertex = true)
    end
    is_vertex && return true

    w = _winding_number(p .- Ref(k), atol) # Winding number of polygon wrt k
    return !(abs(w) < atol) # Returns false is w ≈ 0
end

function _winding_number(v, atol = 1e-12)
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

function get_cells(x, y, A::AbstractMatrix, level::Real = 0.0, mask = [])
    xax, yax = axes(A)
    cells = OrderedDict{Tuple{Int, Int}, UInt8}()
    border_cells = OrderedDict{Tuple{Int, Int}, UInt8}()

    @inbounds for i in first(xax):last(xax)-1
        for j in first(yax):last(yax)-1
            # if any(isnan.([A[i, j], A[i+1, j], A[i+1,j+1], A[i,j+1]]))
            #     continue
            # end

            intersect = (A[i, j] > level) ? 0x01 : 0x00
            (A[i + 1, j] > level) && (intersect |= 0x02)
            (A[i + 1, j + 1] > level) && (intersect |= 0x04)
            (A[i, j + 1] > level) && (intersect |= 0x08)

            if !(intersect == 0x00) && !(intersect == 0x0f)
                if length(mask) > 2 # Check for intersection with masking polygon
                    cell = [[x[i],y[i+1]], [x[i+1],y[i]], [x[i+1], y[i+1]], [x[i], y[i+1]]]
                    inside = in_polygon.(cell, Ref(mask))
                                            
                    if 0 < sum(inside) < 3
                        border_cells[(i,j)] = crossing_lookup[intersect]
                        continue
                    end
                end

                cells[(i,j)] = crossing_lookup[intersect]
            end

        end
    end 

    return cells, border_cells
end

shift = [(0,1), (0,-1), (1,0), (-1,0)] # Next cell to check given crossing
new_edge = (B,T,L,R)
function get_next_cell(edge, index)
    i = trailing_zeros(edge) + 1
    index = index .+ shift[i]
    return new_edge[i], index
end

function find_contour(x, y, A::AbstractMatrix, level::Real = 0.0; mask = [])
    bundle = IsolineBundle(Isoline[], level)
    if level > maximum(A[.!isnan.(A)]) || level < minimum(A[.!isnan.(A)])
        return bundle # No contours to be found
    end

    xax, yax = axes(A) # Valid indices for iterating over
    is = first(xax):last(xax)-1
    js = first(yax):last(yax)-1

    cells, border_cells = get_cells(x, y, A, level, mask)
    
    while Base.length(cells) > 0
        segment = Vector{SVector{2,Float64}}(undef, 0)
        start_index, case = first(cells)
        start_edge = 0x01 << trailing_zeros(case)

        end_index = follow_contour!(cells, border_cells, segment, x, y, A, is, js, start_index, start_edge, level; mask)

        # Check if the contour forms a loop
        isclosed = end_index == start_index ? true : false

        if !isclosed && length(cells) > 0
            # Go back to the starting cell and walk the other direction
            edge, index = get_next_cell(start_edge, start_index)
            !haskey(cells, index) && break 

            follow_contour!(cells, border_cells, reverse!(segment), x, y, A, is, js, index, edge, level)
        end

        seg_length = 0.0
        for i in eachindex(segment)
            i == length(segment) && continue
            seg_length += norm(segment[i + 1] - segment[i])
        end
        isclosed && (seg_length += norm(last(segment) - first(segment)))

        push!(bundle.isolines, Isoline(segment, isclosed, seg_length))
    end
    
    return bundle
end

function intersection(a1, v1, a2, v2)
    V = hcat(v1, -v2)
    det(V) == 0 && return [NaN, NaN] # No intersection

    t = inv(V) * (a2 - a1)
    return a1 + t[1] * v1
end

function param_intersection(a1, v1, a2, v2)
    V = hcat(v1, -v2)
    det(V) == 0 && return [NaN, NaN] # No intersection

    t = inv(V) * (a2 - a1)
    return a1 + t[1] * v1, t[1]
end

function follow_contour!(cells, border_cells, contour, x, y, A, is, js, start_index, start_edge, level; mask = [])
    index = start_index
    edge = start_edge

    push!(contour, get_crossing(x, y, A, index, edge, level))
    while true

        edge = pop!(cells, index) ⊻ edge
        push!(contour, get_crossing(x, y, A, index, edge, level))
        edge, index = get_next_cell(edge, index)


        # if index ∈ keys(border_cells)
        #     edge = pop!(border_cells, index) ⊻ edge
            
        #     # p = get_crossing(x, y, A, index, edge, level) 
        #     # for i in eachindex(mask)
        #     #     @show p - contour[end]
        #     #     ip = mod(i, length(mask)) + 1
        #     #     q, t = param_intersection(mask[i], mask[ip] - mask[i], contour[end], p - contour[end])
        #     #     @show q, t
        #     # end
        #     # boundary_intersection = intersection(contour[end], p - contour[end])
        #     break
        # end
        if index ∈ keys(border_cells)
            @show show index
        end


        (!(index[1] ∈ is) || !(index[2] ∈ js) || index == start_index || index ∈ keys(border_cells)) && break
    end

    return index
end

function get_crossing(x, y, A, index, edge, level)
    i, j = index
    if edge == 0x08 # Left
        xcoord = x[i]
        ycoord = y[j] + (level - A[i,j]) * (y[j + 1] - y[j]) / (A[i, j + 1] - A[i,j])
    elseif edge == 0x01 # Top
        xcoord = x[i] + (level - A[i, j + 1]) * (x[i + 1] - x[i]) / (A[i + 1, j + 1] - A[i, j + 1])
        ycoord = y[j + 1]
    elseif edge == 0x04 # Right
        xcoord = x[i + 1]
        ycoord = y[j] + (level - A[i + 1, j]) * (y[j + 1] - y[j]) / (A[i + 1, j + 1] - A[i + 1, j])
    elseif edge == 0x02 # Bottom
        xcoord = x[i] + (level - A[i,j]) * (x[i + 1] - x[i]) / (A[i + 1, j] - A[i,j])
        ycoord = y[j]
    end

    return SVector(xcoord, ycoord)
end

function contours(x, y, A, levels; mask = [])
    bundles = Vector{IsolineBundle}(undef, Base.length(levels))
    for i in eachindex(levels)
        bundles[i] = find_contour(x, y, A, levels[i]; mask)
    end
    
    return bundles
end

function get_bounding_box(points)
    min_x = minimum(first.(points))
    max_x = maximum(first.(points))

    min_y = minimum(last.(points))
    max_y = maximum(last.(points))

    return ((min_x, max_x), (min_y, max_y))
end

function contour_intersection(a, v, iso::Isoline, i = 1)
    imax = lastindex(iso.points)
    oldsign = 0 # Flag to indicate first iteration
    area = 0.0
    while i <= imax
        w = iso.points[i] - a 
        area = w[1] * v[2] - w[2]*v[1]
        newsign = sign(area) # Sign of cross product area
        if oldsign != 0 && newsign*oldsign < 0 
            i -= 1
            break
        end

        oldsign = newsign
        i += 1
    end

    if i == imax + 1
        θmax = atan(abs(area), abs(dot(v, iso.points[imax] - a)))
        θ1 = atan(abs((iso.points[1][1] - a[1]) * v[2] - (iso.points[1][2] - a[2])*v[1]), abs(dot(v, iso.points[1] - a)))
        
        if abs(θmax) < abs(θ1)
            return iso.points[imax], (imax, imax)
        else
            return iso.points[1], (1,1)
        end 
    else
        p1 = iso.points[i]; p2 = iso.points[i+1]
        return _intersection(a, v, p1, p2-p1), (i, i+1)
    end
end

function _intersection(a1, v1, a2, v2)
    V = hcat(v1, -v2)
    det(V) == 0 && return [NaN, NaN] # No intersection

    t = inv(V) * (a2 - a1)
    return SVector{2}(a1 + t[1] * v1)
end

end # module MarchingSquares
