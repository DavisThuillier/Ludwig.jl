###
### Convex polytope culling
###

"""
    HalfSpace{D}

An oriented half-space `{x : normal·x ≤ offset}` with outward unit `normal`. A convex polytope is the
intersection of its faces' half-spaces.
"""
struct HalfSpace{D}
    normal::SVector{D,Float64}   # outward unit normal
    offset::Float64              # inside ⇔ normal·x ≤ offset
end

"""
    ConvexPolytope{D}

A convex polytope (polygon in 2D, polyhedron in 3D) as the intersection of its bounding `faces`.
Construct from an unordered set of vertices with `ConvexPolytope(vertices)`.
"""
struct ConvexPolytope{D}
    faces::Vector{HalfSpace{D}}
end

"""
    in_convex_polytope(p, polytope; atol=0.0)

Return whether `p` lies inside or on the boundary of `polytope`, within `atol`.
"""
@inline function in_convex_polytope(p::SVector{D,Float64}, polytope::ConvexPolytope{D};
                                    atol::Float64 = 0.0) where {D}
    @inbounds for face in polytope.faces
        dot(face.normal, p) - face.offset > atol && return false
    end
    return true
end

###
### Vertex enumeration
###

"""
    vertices(polytope::ConvexPolytope{D}) -> Vector{SVector{D,Float64}}

Enumerate the corner points of `polytope`: the points where `D` of its face hyperplanes meet and
which satisfy every face's half-space. This is the dual of the half-space representation. Redundant
faces (those interior to the hull) never contribute a corner.
"""
function vertices(polytope::ConvexPolytope{2})
    faces = polytope.faces
    scale = maximum(abs(face.offset) for face in faces; init = 1.0)
    tolerance = 1e-9 * max(scale, 1.0)
    found = SVector{2,Float64}[]
    face_count = length(faces)
    for i in 1:face_count-1, j in i+1:face_count
        normal_matrix = SMatrix{2,2,Float64}(faces[i].normal[1], faces[j].normal[1],
                                             faces[i].normal[2], faces[j].normal[2])
        abs(det(normal_matrix)) ≤ 1e-12 && continue   # parallel face normals: no intersection
        corner = normal_matrix \ SVector(faces[i].offset, faces[j].offset)
        is_feasible(corner, faces, tolerance) || continue
        add_unique_vertex!(found, corner, tolerance)
    end
    return found
end

function vertices(polytope::ConvexPolytope{3})
    faces = polytope.faces
    scale = maximum(abs(face.offset) for face in faces; init = 1.0)
    tolerance = 1e-9 * max(scale, 1.0)
    found = SVector{3,Float64}[]
    face_count = length(faces)
    for i in 1:face_count-2, j in i+1:face_count-1, k in j+1:face_count
        normal_matrix = SMatrix{3,3,Float64}(faces[i].normal[1], faces[j].normal[1], faces[k].normal[1],
                                             faces[i].normal[2], faces[j].normal[2], faces[k].normal[2],
                                             faces[i].normal[3], faces[j].normal[3], faces[k].normal[3])
        abs(det(normal_matrix)) ≤ 1e-12 && continue   # coplanar/parallel normals: no unique meet
        corner = normal_matrix \ SVector(faces[i].offset, faces[j].offset, faces[k].offset)
        is_feasible(corner, faces, tolerance) || continue
        add_unique_vertex!(found, corner, tolerance)
    end
    return found
end

# Whether `point` lies inside every face's half-space, within `tolerance`.
@inline function is_feasible(point::SVector{D,Float64}, faces::Vector{HalfSpace{D}}, tolerance::Float64) where {D}
    for face in faces
        dot(face.normal, point) - face.offset > tolerance && return false
    end
    return true
end

# Append `point` unless a corner within `tolerance` is already present.
function add_unique_vertex!(found::Vector{SVector{D,Float64}}, point::SVector{D,Float64}, tolerance::Float64) where {D}
    for existing in found
        norm(existing - point) ≤ tolerance && return
    end
    push!(found, point)
    return
end

"""
    prune_redundant_faces(polytope::ConvexPolytope{D}) -> ConvexPolytope{D}

Return an equivalent polytope keeping only the faces that actually bound it: a face is retained when
at least `D` of the polytope's vertices lie on its hyperplane (an edge in 2D, a facet in 3D).
"""
function prune_redundant_faces(polytope::ConvexPolytope{D}) where {D}
    corners = vertices(polytope)
    scale = maximum(abs(face.offset) for face in polytope.faces; init = 1.0)
    tolerance = 1e-9 * max(scale, 1.0)
    kept = HalfSpace{D}[]
    for face in polytope.faces
        incident = count(corner -> abs(dot(face.normal, corner) - face.offset) ≤ tolerance, corners)
        incident ≥ D && push!(kept, face)
    end
    return ConvexPolytope{D}(kept)
end

"""
    axis_aligned_bounding_box(polytope::ConvexPolytope{D}) -> NTuple{D,Tuple{Float64,Float64}}

The tightest axis-aligned box containing `polytope`, as a `(min, max)` extent per axis.
"""
function axis_aligned_bounding_box(polytope::ConvexPolytope{D}) where {D}
    corners = vertices(polytope)
    lower = reduce((a, b) -> min.(a, b), corners)
    upper = reduce((a, b) -> max.(a, b), corners)
    return ntuple(d -> (lower[d], upper[d]), D)
end
