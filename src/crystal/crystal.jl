struct Crystal{D, T, L}
    lattice::Lattice{D, T, L}
    point_group::PointGroup{D, T, L}
    irreducible_brillouin_zone::IrreducibleBrillouinZone{D, T}
    # function Crystal(lattice::Latice{D, T, L}, point_group::PointGroup{D}) where {D,T,L}
    #     new{D,T,L}(lattice, point_group, IrreducibleBrillouinZone(l, pg))
    # end
end

lattice(c::Crystal) = c.lattice
point_group(c::Crystal) = c.point_group
irreducible_brillouin_zone(c::Crystal) = c.irreducible_brillouin_zone
brillouin_zone(c::Crystal) = brillouin_zone(lattice(c))

function Crystal(lattice::Lattice{D,T,L}, generators::AbstractVector{AbstractMatrix{T}}) where {D,T,L}
    point_group = PointGroup(generators, lattice)
    return Crystal(lattice, point_group)
end
