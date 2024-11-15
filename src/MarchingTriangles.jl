module MarchingTriangles

using ..Ludwig: Lattices
using StaticArrays
using LinearAlgebra

export lattice_to_mesh, ibz_mesh

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

mutable struct TriangleMesh{T}
    points::Vector{SVector{2, T}}
    edges::Set{Tuple{Int, Int}}
end

TriangleMesh(points::Vector{Vector{Float64}}, edges) = TriangleMesh(map(x -> SVector{2}(x), points), edges) 

function get_neighbors(mesh, i)
    neighbors = Set(Int[])
    for edge in mesh.edges
        if i ∈ edge
            push!(neighbors, setdiff(edge, i)...)
        end
    end
    return neighbors
end

function dfs!(paths, path, visited, mesh, depth, v, start)
    visited[v] = 1
    neighbors = get_neighbors(mesh, v)

    if depth == 1   
        visited[v] = 0 # Clear up vertex to be revisited
        if start ∈ neighbors
            push!(paths, Set(path))
            return nothing
        end
    else
        for j in neighbors
            visited[j] == 0 && dfs!(paths, union(path, j), visited, mesh, depth - 1, j, start)
        end
    end   
    visited[v] = 0 
end

orientation(v1, v2) = v1[1]*v2[2] - v2[1]*v1[2] > 0 ? 1 : -1

function get_triangles(mesh::TriangleMesh, E, level = (maximum(E) + minimum(E)) / 2.0, Δε = (maximum(E) - minimum(E))/2.0)
    visited = zeros(UInt8, length(mesh.points))

    triangles = Set(Vector{Set{Int}}(undef, 0))
    for i in eachindex(mesh.points)
        if abs(E[i] - level) < Δε
            dfs!(triangles, Set([i]), visited, mesh, 3, i, i)
        end
    end

    oriented_triangles = Vector{NTuple{3, Int}}(undef, length(triangles))
    for (i, Δ) in enumerate(triangles)
        j, k, l = Δ
        p1, p2, p3 = map(x -> mesh.points[x], (j,k,l))
        
        if orientation(p2 - p1, p3 - p1) == 1
            oriented_triangles[i] = (j,k,l)
        else
            oriented_triangles[i] = (j,l,k) 
        end
    end
    
    return oriented_triangles
end

function _polygon_mesh(p, basis, N)
    T = inv(basis)

    p_prime = Ref(T) .* p # Transform to orthogonal basis
    x_range, y_range = get_bounding_box(p_prime)

    x_step = (x_range[2] - x_range[1]) / (N -1)
    y_step = (y_range[2] - y_range[1]) / (N -1)

    # Find all points in polygon
    vertices = Tuple{Int, Int}[]
    points = SVector{2,Float64}[]
    for j in 1:N
        for i in 1:N
            k = SVector{2}([x_range[1] + x_step * (i - 1), y_range[1] + y_step * (j - 1)])
            if Lattices.in_polygon(k, p_prime)
                push!(vertices, (i, j))
                push!(points, basis * k)
            end
        end 
    end

    edge_candidates(i,j) = [(i-1, j), (i-1, j+1), (i, j+1), (i, j-1), (i+1,j-1), (i+1, j)]

    edges = Set(Tuple{Int, Int}[])
    for i in eachindex(vertices)
        w = vertices[i]
        border_case = 0b0
        for (j, v) in enumerate(edge_candidates(w...))
            k = findfirst(==(v), vertices)
            if isnothing(k) 
                border_case |= 2^(6-j)
            else
                push!(edges, (i, k)) 
            end
        end

        if border_case == 0b000111
            for v ∈ [(w[1] + 1, w[2] + 1), (w[1] - 1, w[2] - 1)]
                if v ∈ vertices
                    k = findfirst(==((v)), vertices)
                    push!(edges, (i, k))
                else
                    v = (v[1] + 1, v[2])
                    while v ∈ vertices
                        k = findfirst(==((v)), vertices)
                        push!(edges, (i, k))
                        v = (v[1] + 1, v[2] + 1)
                    end
                end
            end
        end
    end

    return TriangleMesh(points, edges)
end

function bz_mesh(l::Lattices.Lattice, N)
    rlv = Lattices.reciprocal_lattice_vectors(l)
    bz = Lattices.get_bz(l)

    mesh = _polygon_mesh(bz, rlv, N)
    return mesh
end

function ibz_mesh(l::Lattices.Lattice, N)
    rlv = Lattices.reciprocal_lattice_vectors(l)
    ibz = Lattices.get_ibz(l)

    basis = Matrix{Float64}(undef, 2, 0)
    for i in eachindex(ibz)
        j = i == length(ibz) ? 1 : i + 1

        if ibz[i] == [0.0, 0.0] || ibz[j] == [0.0, 0.0]
            basis = hcat(basis, ibz[i] - ibz[j])
        end
    end

    mesh = _polygon_mesh(ibz, basis, N)
    return mesh
end

function get_bounding_box(points)
    min_x = minimum(first.(points))
    max_x = maximum(first.(points))

    min_y = minimum(last.(points))
    max_y = maximum(last.(points))

    return ((min_x, max_x), (min_y, max_y))
end

# First, Second, and Third Edges
#       2
#      / \
#   S /   \ F
#    /     \
#  3 ------- 1
#       T

F, S, T = 0x01, 0x02, 0x04
edges = (F, S, T)
pair_to_edge = Dict((1,2) => 1, (2,1) => 1, (2,3) => 2, (3,2) => 2, (3,1) => 3, (1,3) => 3)
crossing_lut = [F|T, F|S, S|T, S|T, F|S, F|T]

function get_cells(triangles, A::AbstractVector, level::Real = 0.0)
    cells = Dict{Int, UInt8}()

    for (i, Δ) in enumerate(triangles)
        intersect = A[Δ[1]] > level ? 0x01 : 0x00
        A[Δ[2]] > level && (intersect |= 0x02)
        A[Δ[3]] > level && (intersect |= 0x04)
        if intersect != 0x00 && intersect != 0x07
            cells[i] = crossing_lut[intersect]
        end
    end

    return cells
end

get_next_edge(start_edge, case) = start_edge ⊻ case

function get_next_cell(triangles, edge, index)
    edge_index = trailing_zeros(edge)
    i1 = triangles[index][edge_index + 1]
    i2 = triangles[index][mod(edge_index + 1, 3) + 1]

    for i in keys(triangles) 
        if ((i1, i2) ⊆ triangles[i] && i != index)
            j = i
            j1 = findfirst(==(i1), triangles[j])
            j2 = findfirst(==(i2), triangles[j])
            return edges[pair_to_edge[(j1, j2)]], j 
        end
    end

    return 0x00, -1
end

function get_crossing(vertices, Δ, A, edge, level)
    edge_index = trailing_zeros(edge)
    i1 = Δ[edge_index + 1]
    i2 = Δ[mod(edge_index + 1, 3) + 1]

    p1 = vertices[i1]
    p2 = vertices[i2] 

    p = p1 + (p2 - p1) * (level - A[i1]) / (A[i2] - A[i1]) 

    return SVector{2}(p...)
end

function find_contour(mesh, A, level::Real = 0.0, Δε = (maximum(A) - minimum(A)) / 2.0)
    bundle = IsolineBundle(Isoline[], level)

    vertices = mesh.points
    triangles = get_triangles(mesh, A, level, Δε)

    cells = get_cells(triangles, A, level)
    cell_triangles = Dict{Int, Tuple{Int, Int, Int}}()
    for i in keys(cells)
        cell_triangles[i] = triangles[i]
    end

    while Base.length(cells) > 0
        segment = Vector{SVector{2,Float64}}(undef, 0)
        start_index, case = first(cells)
        start_edge = 0x01 << trailing_zeros(case) 

        end_index = follow_contour!(cells, segment, vertices, triangles, cell_triangles, A, start_index, start_edge, level)

        # Check if the contour forms a loop
        isclosed = end_index == start_index ? true : false

        if !isclosed && Base.length(cells) > 0
            # Go back to the starting cell and walk the other direction
            edge, index = get_next_cell(cell_triangles, start_edge, start_index)
            !haskey(cells, index) && break

            follow_contour!(cells, reverse!(segment), vertices, triangles, cell_triangles, A, index, edge, level)
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

# find_contour(mesh, A, level::Real = 0.0, Δε = (maximum(A) - minimum(A)) / 2.0) = contours(mesh, A, [level], Δε)

function follow_contour!(cells, contour, vertices, triangles, cell_triangles, A, start_index, start_edge, level)
    index = start_index
    edge  = start_edge
    
    push!(contour, get_crossing(vertices, triangles[index], A, edge, level))
    while true
        edge = pop!(cells, index) ⊻ edge
        push!(contour, get_crossing(vertices, triangles[index], A, edge, level))
        edge, index = get_next_cell(cell_triangles, edge, index)

        (!haskey(cells, index) || index == start_index) && break
    end

    return index
end

function contours(mesh, A, levels, Δε)
    bundles = Vector{IsolineBundle}(undef, Base.length(levels))
    
    for i in eachindex(levels)
        bundles[i] = find_contour(mesh, A, levels[i], Δε)
    end
    
    return bundles
end

end # module