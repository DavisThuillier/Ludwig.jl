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

function get_cells(A::AbstractMatrix, level::Real = 0.0)
    x, y = axes(A)
    cells = OrderedDict{Tuple{Int, Int}, UInt8}()

    @inbounds for i in first(x):last(y)-1
        for j in first(y):last(y)-1
            intersect = (A[i, j] > level) ? 0x01 : 0x00
            (A[i + 1, j] > level) && (intersect |= 0x02)
            (A[i + 1, j + 1] > level) && (intersect |= 0x04)
            (A[i, j + 1] > level) && (intersect |= 0x08)

            if !(intersect == 0x00) && !(intersect == 0x0f)
                cells[(i,j)] = crossing_lookup[intersect]
            end

        end
    end 

    return cells
end

shift = [(0,1), (0,-1), (1,0), (-1,0)] # Next cell to check given crossing
new_edge = (B,T,L,R)
function get_next_cell(edge, index)
    i = trailing_zeros(edge) + 1
    index = index .+ shift[i]
    return new_edge[i], index
end

function find_contour(x, y, A::AbstractMatrix, level::Real = 0.0)
    bundle = IsolineBundle(Isoline[], level)

    xax, yax = axes(A) # Valid indices for iterating over
    is = first(xax):last(xax)-1
    js = first(yax):last(yax)-1

    cells = get_cells(A, level)

    while Base.length(cells) > 0
        segment = Vector{SVector{2,Float64}}(undef, 0)
        start_index, case = first(cells)
        start_edge = 0x01 << trailing_zeros(case)

        end_index = follow_contour!(cells, segment, x, y, A, is, js, start_index, start_edge, level)

        # Check if the contour forms a loop
        isclosed = end_index == start_index ? true : false

        if !isclosed && length(cells) > 0
            # Go back to the starting cell and walk the other direction
            edge, index = get_next_cell(start_edge, start_index)
            !haskey(cells, index) && break 
            
            follow_contour!(cells, reverse!(segment), x, y, A, is, js, index, edge, level)
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

function follow_contour!(cells, contour, x, y, A, is, js, start_index, start_edge, level)
    index = start_index
    edge = start_edge

    push!(contour, get_crossing(x, y, A, index, edge, level))
    while true

        edge = pop!(cells, index) ⊻ edge
        push!(contour, get_crossing(x, y, A, index, edge, level))
        
        edge, index = get_next_cell(edge, index)


        (!(index[1] ∈ is) || !(index[2] ∈ js) || index == start_index) && (break)
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

function contours(x, y, A, levels)
    bundles = Vector{IsolineBundle}(undef, Base.length(levels))
    for i in eachindex(levels)
        bundles[i] = find_contour(x, y, A, levels[i])
    end
    
    return bundles
end



