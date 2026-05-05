module Ludwig 
    # include("constants.jl")

    using LinearAlgebra
    using StaticArrays

    include(joinpath("crystal", "polytope.jl"))
    include(joinpath("utilities", "convex_hull.jl"))
    include(joinpath("crystal", "brillouin_zone.jl"))
    include(joinpath("crystal", "lattice.jl"))
    include(joinpath("crystal", "point_group.jl"))
    include(joinpath("crystal", "irreducible_brillouin_zone.jl"))
    include(joinpath("crystal", "crystal.jl"))

    export Lattice
    export reciprocal_lattice_vectors

    export BrillouinZone
    export PointGroup
    export IrreducibleBrillouinZone
    export Crystal

    """
        plot_brillouin_zone(bz::BrillouinZone{D}; <keyword arguments>)
        plot_brillouin_zone!(ax, bz::BrillouinZone{D}; <keyword arguments>)

    Plot a Brillouin zone: a polygon for `D = 2`, a polyhedral mesh for
    `D = 3`. The non-mutating form creates a new figure with an `Axis` (2D)
    or `Axis3` (3D); the mutating form draws into an existing `ax`. Methods
    provided by the `LudwigMakieExt` extension; activate when a Makie backend
    (e.g. `GLMakie`, `CairoMakie`, `WGLMakie`) is loaded.
    """
    function plot_brillouin_zone end
    function plot_brillouin_zone! end
    export plot_brillouin_zone, plot_brillouin_zone!

    """
        plot_irreducible_brillouin_zone(ibz::IrreducibleBrillouinZone{D}; <keyword arguments>)
        plot_irreducible_brillouin_zone!(ax, ibz::IrreducibleBrillouinZone{D}; <keyword arguments>)

    Plot an irreducible Brillouin zone: a polygon for `D = 2`, a polyhedral
    mesh for `D = 3`. The non-mutating form creates a new figure with an
    `Axis` (2D) or `Axis3` (3D); the mutating form draws into an existing
    `ax`. Methods provided by the `LudwigMakieExt` extension; activate when a
    Makie backend (e.g. `GLMakie`, `CairoMakie`, `WGLMakie`) is loaded.
    """
    function plot_irreducible_brillouin_zone end
    function plot_irreducible_brillouin_zone! end
    export plot_irreducible_brillouin_zone, plot_irreducible_brillouin_zone!

end
