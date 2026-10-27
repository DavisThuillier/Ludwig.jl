
"""
    SimplexMesh{D}

Mesh constituted of 'D' - 1 dimensional simplices embedded in 'D' dimensions.
"""
struct SimplexMesh{D}
    vertices::Vector{SVector{D,Float64}}
    simplices::Vector{NTuple{D,Int}}
end

vertices(mesh::SimplexMesh) = mesh.vertices
simplices(mesh::SimplexMesh) = mesh.simplices
centroids(mesh::SimplexMesh{D}) where {D} = [sum([mesh.vertices[i] for i ∈  simplex]) / D for simplex ∈ mesh.simplices] 

"""
    build_simplex(i::Int, mesh::SimplexMesh)

Collect vertices corresponding to simplex 'i' from 'mesh'.
"""
function build_simplex(i::Int, mesh::SimplexMesh)
    simplex = mesh.simplices[i]
    return map(j -> mesh.vertices[j], simplex)
end

build_simplices(mesh::SimplexMesh) = [build_simplex(i, mesh) for i ∈ eachindex(mesh.simplices)]

"""
    volume(i::Int, mesh::SimplexMesh{2})
    volume(i::Int, mesh::SimplexMesh{3})

Compute the volume of simplex 'i' from 'mesh'.
"""
function volume(i::Int, mesh::SimplexMesh{2})
    vertex_a, vertex_b = build_simplex(i, mesh)
    return norm(vertex_b - vertex_a)
end

function volume(i::Int, mesh::SimplexMesh{3})
    vertex_a, vertex_b, vertex_c = build_simplex(i, mesh)
    return norm(cross(vertex_b - vertex_a, vertex_c - vertex_a)) / 2
end

"""
    perimeter(mesh::SimplexMesh{2})
    perimeter(mesh::SimplexMesh{3}, ibz::ConvexPolytope{3}; atol = 1e-9·max(offset, 1))

Total length of the boundary of `mesh`. For a `SimplexMesh{3}` surface this is the summed length of the
edges lying on the boundary of `ibz` — the seam where a `level_set` surface was clipped to the `ibz` mask.
An edge is on the seam when both of its endpoints sit on a *common* face of `ibz` within `atol`; requiring a
shared face rather than each endpoint merely touching some face excludes a chord that spans two faces near an
`ibz` corner. For a `SimplexMesh{2}` contour the boundary is a measure-zero set of points, so it is zero.

`atol` defaults to the polytope tolerance and may be widened to match a mesh welded with a coarser tolerance.
The test is independent of whether `mesh` is vertex-welded.
"""
perimeter(::SimplexMesh{2}, ::Any = nothing) = 0.0

function perimeter(mesh::SimplexMesh{3}, ibz::ConvexPolytope{3};
                   atol::Float64 = 1e-9 * max(maximum(abs(face.offset) for face in ibz.faces; init = 1.0), 1.0))
    faces = prune_redundant_faces(ibz).faces
    verts = mesh.vertices
    total = 0.0
    for (i, j, k) in mesh.simplices
        endpoints_share_face(verts[i], verts[j], faces, atol) && (total += norm(verts[i] - verts[j]))
        endpoints_share_face(verts[j], verts[k], faces, atol) && (total += norm(verts[j] - verts[k]))
        endpoints_share_face(verts[k], verts[i], faces, atol) && (total += norm(verts[k] - verts[i]))
    end
    return total
end

# Whether both edge endpoints lie on a common face plane of the polytope within `atol`.
function endpoints_share_face(endpoint_a::SVector{3,Float64}, endpoint_b::SVector{3,Float64},
                              faces::Vector{HalfSpace{3}}, atol::Float64)
    for face in faces
        if abs(dot(face.normal, endpoint_a) - face.offset) ≤ atol &&
           abs(dot(face.normal, endpoint_b) - face.offset) ≤ atol
            return true
        end
    end
    return false
end

"""
    integrate_over_mesh(g, mesh::SimplexMesh{D}; threaded=false, nchunks=Threads.nthreads())

Integrate `g` over `mesh` by summing, simplex by simplex, the integral of the linear interpolant of `g`
through its nodal values at the simplex vertices.

Set `threaded=true` to split the simplices into `nchunks` contiguous blocks summed concurrently (one task
per block, partials reduced on the caller). `g` must then be thread-safe, and the reduction order changes,
so floating-point results may differ from the serial path in the last bits.
"""
function integrate_over_mesh(g, mesh::SimplexMesh{D};
                             threaded::Bool = false,
                             nchunks::Int = Threads.nthreads()) where {D}
    simplex_count = length(mesh.simplices)
    simplex_count == 0 && return 0.0
    threaded || return integrate_over_mesh_subset(g, mesh, 1, simplex_count)
    nchunks = clamp(nchunks, 1, simplex_count)
    tasks = map(1:nchunks) do c
        first_simplex = ((c - 1) * simplex_count) ÷ nchunks + 1
        last_simplex = (c * simplex_count) ÷ nchunks
        Threads.@spawn integrate_over_mesh_subset(g, mesh, first_simplex, last_simplex)
    end
    return sum(fetch, tasks; init = 0.0)
end

"""
    integrate_over_mesh_subset(g, mesh::SimplexMesh, first_simplex::Int, last_simplex::Int)

Sum the linear-interpolant integral of `g` over simplices `first_simplex:last_simplex`, accumulating into
a single task-local total.
"""
function integrate_over_mesh_subset(g::Function, mesh::SimplexMesh{D}, first_simplex::Int, last_simplex::Int) where {D}
    test_point = rand(D)
    total = zero(typeof(g(test_point)))
    for i ∈ first_simplex:last_simplex
        total += integrate_over_simplex(g, i, mesh)
    end
    return total
end

function integrate_over_mesh_subset(ϕ::AbstractVector, mesh::SimplexMesh{D}, first_simplex::Int, last_simplex::Int) where {D}
    total = zero(eltype(ϕ))
    for i ∈ first_simplex:last_simplex
        total += integrate_over_simplex(ϕ, i, mesh)
    end
    return total
end

"""
    integrate_over_simplex(g, i::Int, mesh::SimplexMesh{2})
    integrate_over_simplex(g, i::Int, mesh::SimplexMesh{3})

Integrate the linear interpolant of `g` over simplex 'i': its `(D-1)`-volume times the mean of `g` over the
simplex vertices (exact for the interpolant — the trapezoid rule on a segment, the centroid rule on a triangle).
"""
function integrate_over_simplex(g::Function, i::Int, mesh::SimplexMesh{2})
    a, b = build_simplex(i, mesh)
    return volume(i, mesh) * (g(a) + g(b)) / 2
end

function integrate_over_simplex(g::Function, i::Int, mesh::SimplexMesh{3})
    a, b, c = build_simplex(i, mesh)
    return volume(i, mesh) * (g(a) + g(b) + g(c)) / 3
end

function integrate_over_simplex(ϕ::AbstractVector, i::Int, mesh::SimplexMesh{2})
    si, sj = mesh.simplices[i]
    return volume(i, mesh) * (ϕ[si] + ϕ[sj]) / 2
end

function integrate_over_simplex(ϕ::AbstractVector, i::Int, mesh::SimplexMesh{3})
    si, sj, sk = mesh.simplices[i]
    return volume(i, mesh) * (ϕ[si] + ϕ[sj] + ϕ[sk]) / 3
end
