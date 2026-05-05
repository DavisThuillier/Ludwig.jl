struct Crystal{D, T, L}
    lattice::Lattice{D, T, L}
    point_group::PointGroup{D, T, L}
    irreducible_brillouin_zone::IrreducibleBrillouinZone{D, T}
end

function Crystal(lattice::Lattice{D, T, L}, pg::PointGroup{D, T, L}) where {D, T, L}
    return Crystal{D, T, L}(lattice, pg, IrreducibleBrillouinZone(lattice, pg))
end

function Crystal(lattice::Lattice{D, T, L},
                 generators::AbstractVector{<:AbstractMatrix}) where {D, T, L}
    return Crystal(lattice, PointGroup(generators, lattice))
end

lattice(c::Crystal) = c.lattice
point_group(c::Crystal) = c.point_group
irreducible_brillouin_zone(c::Crystal) = c.irreducible_brillouin_zone
brillouin_zone(c::Crystal) = c.lattice.brillouin_zone
