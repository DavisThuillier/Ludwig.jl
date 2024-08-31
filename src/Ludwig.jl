module Ludwig

    "Conductance quantum in Siemens"
    const G0::Float64 = 7.748091729e-5

    "Boltzmann constant in eV/K"
    const kb::Float64 = 8.617333262e-5

    "Reduced Planck's constant in eV.s"
    const hbar::Float64 = 6.582119569e-16

    "Electron charge in C"
    const e_charge::Float64 = 1.602176634e-19

    export G0, kb, hbar, e_charge # Physical constants
    export f0

    import StaticArrays: SVector, MVector, MMatrix, SMatrix
    using LinearAlgebra
    using ForwardDiff
    import DataStructures: OrderedDict
    using ProgressBars
    using Interpolations
    using IterativeSolvers

    """
        f0(E, T)

    Return the value of the Fermi-Dirac distribution for energy `E` and temperature `T`.

    ```math
        f^{(0)}(\\varepsilon) = \\frac{1}{1 + e^{\\varepsilon/k_B T}}
    ```
    """
    f0(E::Float64, T::Float64) = 1 / (exp(E/T) + 1)

    """
        map_to_first_bz(k)
    Map a vector `k` to the ``d``-dimensional centered unit cube where ``d`` is the dimension of `k`.
    """
    map_to_first_bz(k) = mod.(k .+ 0.5, 1.0) .- 0.5

    include("mesh/marching_squares.jl")
    include("mesh/mesh.jl")
    include("integration.jl")
    include("properties.jl")
    include("utilities.jl")
    include("vertex_factors.jl")

    
end # module Ludwig
