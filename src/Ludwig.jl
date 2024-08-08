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

    include("./mesh/marching_squares.jl")
    include("./mesh/mesh.jl")
    include("./integration.jl")
    include("./properties.jl")
    include("./form_factors.jl")

    function Weff_squared_123(p1::Patch, p2::Patch, p3::Patch, Fpp::Function, Fpk::Function, k4, μ4)
        f13 = Fpp(p1, p3) 
        f23 = Fpp(p2, p3)
        f14 = Fpk(p1, k4, μ4)
        f24 = Fpk(p2, k4, μ4)

        return abs2(f13*f24 - f14*f23) + 2 * abs2(f14*f23)

    end

    function Weff_squared_124(p1::Patch, p2::Patch, p4::Patch, Fpp::Function, Fpk::Function, k3, μ3)
        f14 = Fpp(p1, p4)
        f24 = Fpp(p2, p4)

        f13 = Fpk(p1, k3, μ3) 
        f23 = Fpk(p2, k3, μ3)

        return abs2(f13*f24 - f14*f23) + 2 * abs2(f14*f23)
    end

    """
        get_bands(H, N)

    Return an interpolation of the eigenvalues of `H` on a square grid [-0.5, 0.5].

    It is assumed that `H` is a function of a vector of length 2 and returns a square matrix.
    `N` is the number of points between -0.5 and 0.5 used for interpolation.
    """
    function get_bands(H::Function, N::Int)
        n_bands = size(H([0.0, 0.0]))[1] # Number of bands
        E = Array{Float64}(undef, N, N, n_bands)
        x = LinRange(-0.5, 0.5, N)
        for i in 1:N, j in 1:N
            E[i, j, :] .= eigvals(H([x[i], x[j]]))# Get eigenvalues (bands) of each k-point
        end
    
        sitps = Vector{ScaledInterpolation}(undef, n_bands)
        for i in 1:n_bands
            itp = interpolate(E[:,:,i], BSpline(Cubic(Line(OnGrid()))))
            sitps[i] = scale(itp, -0.5:1/(N-1):0.5, -0.5:1/(N-1):0.5)
        end
    
        return sitps
    end

end # module Ludwig
