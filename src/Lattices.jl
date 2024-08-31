module Lattices2D

abstract type Lattice end

primitives(l::Lattice) = l.primitives

struct RectangularLattice <: Lattice
    primitives::Matrix
    RectangularLattice(a::Real, b::Real) = a == b ? SquareLattice(a) : new([a 0.0; 0.0 b])
end

struct SquareLattice <: Lattice
    primitives::Matrix
end

SquareLattice(a) = SquareLattice([a 0.0; 0.0 a])


end # module
