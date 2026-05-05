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

    export Lattice
    export reciprocal_lattice_vectors

    export BrillouinZone

    """
        plot_brillouin_zone(bz::BrillouinZone{3}; <keyword arguments>)

    Plot a 3D Brillouin zone in a new figure with an `Axis3`. Method provided by
    the `LudwigMakieExt` extension; activates when a Makie backend (e.g.
    `GLMakie`, `CairoMakie`, `WGLMakie`) is loaded.
    """
    function plot_brillouin_zone end
    export plot_brillouin_zone

end
