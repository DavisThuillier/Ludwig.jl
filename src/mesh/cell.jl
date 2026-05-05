struct Cell{D, T, L}
    e::T
    k::SVector{D,T}
    v::SVector{D,T}
    de::T
    dV::T
    jinv::SMatrix{D,D,T,L}
    djinv::Float64
    band::Int
end

Cell(e::T, k::SVector{D, T}, v::SVector{D,T}, de::T, dV::T, jinv::SMatrix{D,D,T,L}) where {D, T, L} = Cell(e, k, v, de, dV, jinv, det(jinv), 1)
Cell(e::T, k::SVector{D, T}, v::SVector{D,T}, de::T, dV::T, jinv::SMatrix{D,D,T,L}, djinv::T) where {D, T, L} = Cell(e, k, v, de, dV, jinv, djinv, 1)
