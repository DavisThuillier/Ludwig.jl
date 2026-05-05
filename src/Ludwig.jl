module Ludwig 
    # include("constants.jl")

    using LinearAlgebra
    using StaticArrays

    include(joinpath("crystal", "polytope.jl"))
    include(joinpath("utilities", "convex_hull.jl"))
    include(joinpath("crystal", "brillouin_zone.jl"))
    include(joinpath("crystal", "irreducible_brillouin_zone.jl"))
    include(joinpath("crystal", "lattice.jl"))
    include(joinpath("crystal", "point_group.jl"))
    include(joinpath("crystal", "crystal.jl"))

end
