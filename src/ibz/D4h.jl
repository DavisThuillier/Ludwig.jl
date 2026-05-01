struct D4h <: PointGroup end

const D4h_OPERATIONS = let T = Float64
    C4 = SMatrix{3, 3, T}([0 -1 0; 1 0 0; 0 0 1])    # C₄ around z
    σh = SMatrix{3, 3, T}([1 0 0; 0 1 0; 0 0 -1])    # mirror in z = 0
    σv = SMatrix{3, 3, T}([1 0 0; 0 -1 0; 0 0 1])    # mirror in y = 0
    close_group([C4, σh, σv])
end

function ibz(::D4h, lattice::NamedTuple)
    a, c = lattice.a, lattice.c
    T = promote_type(typeof(a), typeof(c))

    vertices = SVector{3, T}[
        SVector(zero(T), zero(T), zero(T)),    # Γ
        SVector(T(π)/a,  zero(T), zero(T)),    # X
        SVector(T(π)/a,  T(π)/a,  zero(T)),    # M
        SVector(zero(T), zero(T), T(π)/c ),    # Z
        SVector(T(π)/a,  zero(T), T(π)/c ),    # R
        SVector(T(π)/a,  T(π)/a,  T(π)/c ),    # A
    ]

    edges = [
        (1, 2), (2, 3), (1, 3),    # bottom triangle (k_z = 0)
        (4, 5), (5, 6), (4, 6),    # top triangle    (k_z = π/c)
        (1, 4), (2, 5), (3, 6),    # vertical edges
    ]

    faces = [
        [1, 2, 3],          # k_z = 0   (bottom triangle)
        [4, 5, 6],          # k_z = π/c (top triangle)
        [1, 2, 5, 4],       # k_y = 0
        [2, 3, 6, 5],       # k_x = π/a
        [1, 3, 6, 4],       # k_x = k_y
    ]

    return IBZ(
        vertices       = vertices,
        edges          = edges,
        faces          = faces,
        operations     = D4h_OPERATIONS,
        angular_sector = (zero(T), T(π)/4),
        labels         = [:Γ, :X, :M, :Z, :R, :A],
    )
end
