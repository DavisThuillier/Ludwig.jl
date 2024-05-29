module Ludwig

    const G0::Float64 = 7.748091729e-5 # Conductance quantum in Siemens
    const kb::Float64 = 8.6173e-5 # Boltzmann constant in eV / T
    const hbar::Float64 = 1.054571817e-34 # In J.s 
    export G0, kb, hbar

    export f0
    """
        f0(E, T)

    Return the value of the Fermi-Dirac distribution for energy E and temperature T.
    """
    f0(E::Float64, T::Float64) = 1 / (exp(E/T) + 1)

    import StaticArrays: SVector, MVector
    using LinearAlgebra
    using ForwardDiff
    import DataStructures: OrderedDict
    using ProgressBars
    using Interpolations

    include("./mesh/marching_squares.jl")
    include("./mesh/mesh.jl")
    include("./integration.jl")

end # module Ludwig
