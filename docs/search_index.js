var documenterSearchIndex = {"docs":
[{"location":"mesh/mesh_sb/#Single-Band-Mesh","page":"Single Band Mesh","title":"Single Band Mesh","text":"","category":"section"},{"location":"mesh/mesh_sb/","page":"Single Band Mesh","title":"Single Band Mesh","text":"Ludwig.generate_mesh","category":"page"},{"location":"mesh/mesh_sb/#Ludwig.generate_mesh","page":"Single Band Mesh","title":"Ludwig.generate_mesh","text":"generate_mesh(E::AbstractArray{<:Real, 2}, H, band_index, T, n_levels, n_angles[, α])\n\nGenerate a Mesh of (n_angles - 1) x (n_levels - 1) patches per quadrant using marching squares to find energy contours of E, the energy eigenvalues of H corresponding to band_index evaluated on a regular grid of mathbfk in 00 05times00 05. \n\nThe width of the Fermi tube is pm α T. \n\n\n\n\n\n","category":"function"},{"location":"collision_operator/#Collision-Operator","page":"Collision Operator","title":"Collision Operator","text":"","category":"section"},{"location":"collision_operator/#Single-Band-and-Naïve-Multiband","page":"Collision Operator","title":"Single Band and Naïve Multiband","text":"","category":"section"},{"location":"collision_operator/","page":"Collision Operator","title":"Collision Operator","text":"For a single band with a momentum independent scattering potential V(mathbfq) = V,","category":"page"},{"location":"collision_operator/","page":"Collision Operator","title":"Collision Operator","text":"    mathbfL_ij = frac1d A_i frac2pi1 - f^(0)_i V^2 frac1(2pi)^6 sum_m (f^(0)_j (1 - f^(0)_m) mathcalK_ijm - 2f^(0)_m (1 - f^(0)_j)mathcalK_imj) ","category":"page"},{"location":"collision_operator/","page":"Collision Operator","title":"Collision Operator","text":"where","category":"page"},{"location":"collision_operator/","page":"Collision Operator","title":"Collision Operator","text":"    mathcalK_ijm = int_i d^2 mathbfk_i int_j d^2 mathbfk_j int_m d^2 mathbfk_m (1 - f^(0)(mathbfk_i + mathbfk_j - mathbfk_m)) delta(varepsilon_i + varepsilon_j - varepsilon_m - varepsilon(mathbfk_i + mathbfk_j - mathbfk_m))","category":"page"},{"location":"collision_operator/","page":"Collision Operator","title":"Collision Operator","text":"A first attempt at incorporating interband scattering was to perform a sum over the bands for the energy corresponding the final momentum computed by enforcing momentum conservation:","category":"page"},{"location":"collision_operator/","page":"Collision Operator","title":"Collision Operator","text":"Ludwig.Γabc!","category":"page"},{"location":"collision_operator/#Ludwig.Γabc!","page":"Collision Operator","title":"Ludwig.Γabc!","text":"Γabc!(ζ, u, a::Patch, b::Patch, c::Patch, T, εabc, ε::Function)\n\nCompute the integral\n\n    mathcalK_abc equiv int_a int_b int_c (1 - f^(0)(mathbfk_a + mathbfk_b + mathbfk_c)) delta(varepsilon_a + varepsilon_b - varepsilon_c - varepsilon(mathbfk_a + mathbfk_b - mathbfk_c)) \n\nwith dispersion ε at temperature T.\n\n    int_i equiv frac1a^2 int_mathbfk in mathcalP_i d^2mathbfk\n\nis an integral over momenta in patch mathcalP_i. \n\n\n\n\n\nΓabc!(ζ, u, a::Patch, b::Patch, c::Patch, T, k, εabc, itp)\n\nCompute mathcalK^_abc with k given by mathbfk equiv overbarmathbfk_a + overbarmathbfk_b - overbarmathbfk_c using itp as an interpolation of the dispersion.\n\n\n\n\n\n","category":"function"},{"location":"collision_operator/#Improved-Multiband","page":"Collision Operator","title":"Improved Multiband","text":"","category":"section"},{"location":"collision_operator/","page":"Collision Operator","title":"Collision Operator","text":"The above model is not realistic, however. The bands consist of hybridized orbitals which have different overlap. To account for this, the next simplest model one can propose for the scattering term is","category":"page"},{"location":"collision_operator/","page":"Collision Operator","title":"Collision Operator","text":"    Usum_i sum_ab n_ia n_ib","category":"page"},{"location":"collision_operator/","page":"Collision Operator","title":"Collision Operator","text":"where i represents a site index and a b are orbital indices. To handle this perturbation in our framework of scattering using the Born approximation, we need to evaluate this interaction term in the eigenbasis of the bare Hamiltonian.","category":"page"},{"location":"collision_operator/","page":"Collision Operator","title":"Collision Operator","text":"beginaligned\n    sum_i n_ia n_i b = sum_ic^dagger_ia c_ia c^dagger_ib c_ib\n    = frac1N^2 sum_mathbfk_1 mathbfk_2 mathbfk_3 mathbfk_4 left( e^i (mathbfk_3 + mathbfk_4 - mathbfk_1 - mathbfk_2) cdot mathbfR_iright ) c^dagger_mathbfk_3a c_mathbfk_1a c^dagger_mathbfk_4b c_mathbfk_2b\n    = frac1N^2 sum_mathbfk_1 mathbfk_2 mathbfk_3 mathbfk_4 N delta_mathbfk_4mathbfk_1 + mathbfk_2 - mathbfk_3 c^dagger_mathbfk_3a c_mathbfk_1a c^dagger_mathbfk_4b c_mathbfk_2b\n    = frac1N sum_mathbfk_1 mathbfk_2 mathbfq c^dagger_mathbfk_1 - mathbfqa c_mathbfk_1a c^dagger_mathbfk_2 + mathbfqb c_mathbfk_2b\n    = frac1N sum_mathbfk_1 mathbfk_2 mathbfq sum_munusigmatau c^dagger_mathbfk_1 - mathbfqsigma left( W_mathbfk_1 - mathbfq^a sigmaright)^* W_mathbfk_1^a mu c_mathbfk_1 mu c^dagger_mathbfk_2 + mathbfqtau left( W_mathbfk_2 + mathbfq^b tauright)^* W_mathbfk_2^b nu c_mathbfk_2 nu\nendaligned","category":"page"},{"location":"collision_operator/","page":"Collision Operator","title":"Collision Operator","text":"From the above, we can define the interband connection as","category":"page"},{"location":"collision_operator/","page":"Collision Operator","title":"Collision Operator","text":"    F_mathbfk_1 mathbfk_2^munu equiv sum_a left(W_mathbfk_1^a mu right)^* W_mathbfk_2^a nu = sum_a left(W^dagger_mathbfk_1right)^mu a W_mathbfk_2^a nu  = left(W^dagger_mathbfk_1 W_mathbfk_2 right)^mu nu  ","category":"page"},{"location":"collision_operator/","page":"Collision Operator","title":"Collision Operator","text":"so the interaction term becomes","category":"page"},{"location":"collision_operator/","page":"Collision Operator","title":"Collision Operator","text":"Usum_i sum_ab n_ia n_ib = fracUN sum_mathbfk_1 mathbfk_2 mathbfq sum_munusigmatau c^dagger_mathbfk_1 - mathbfqsigma  F^sigmamu_mathbfk_1 - mathbfq mathbfk_1  c_mathbfk_1 mu c^dagger_mathbfk_2 + mathbfqtau F^taunu_mathbfk_2 + mathbfq mathbfk_2 c_mathbfk_2 nu","category":"page"},{"location":"collision_operator/","page":"Collision Operator","title":"Collision Operator","text":"The collision matrix elements can then be populated using ","category":"page"},{"location":"collision_operator/","page":"Collision Operator","title":"Collision Operator","text":"Ludwig.electron_electron","category":"page"},{"location":"collision_operator/#Ludwig.electron_electron","page":"Collision Operator","title":"Ludwig.electron_electron","text":"electron_electron(grid::Vector{Patch}, f0s::Vector{Float64}, i::Int, j::Int, itps::Vector{ScaledInterpolation}, T::Real, Fpp::Function, Fpk::Function)\n\nCompute the element (i,j) of the linearized Boltzmann collision operator for electron electron scattering.\n\nThe bands used to construct grid are callable using the interpolated dispersion in itps. The vector f0s stores the value of the Fermi-Dirac distribution at each patch center an can be calculated independent of i and j. The functions Fpp and Fpk are vertex factors that provide the effective, spin-summed transition rate as\n\n    W^2_texteff = U^2 left(   F^mu_1mu_3_mathbfk_1mathbfk_3 F^mu_2mu_4_mathbfk_2mathbfk_4 -  F^mu_1mu_4_k_1k_4 F^mu_2mu_3_mathbfk_2mathbfk_3 ^2 + 2   F^mu_1mu_4_mathbfk_1mathbfk_4 F^mu_2mu_3_mathbfk_2mathbfk_3 ^2 \n\n\n\n\n\n","category":"function"},{"location":"mesh/mesh_mb/#Multiband-Mesh","page":"Multiband Meshes","title":"Multiband Mesh","text":"","category":"section"},{"location":"mesh/mesh_mb/","page":"Multiband Meshes","title":"Multiband Meshes","text":"Ludwig.multiband_mesh","category":"page"},{"location":"mesh/mesh_mb/#Ludwig.multiband_mesh","page":"Multiband Meshes","title":"Ludwig.multiband_mesh","text":"multiband_mesh(bands, W, T, n_levels, n_angles[, N, α])\n\nGenerate a Mesh of (n_angles - 1) x (n_levels - 1) patches per quadrant per band centered on each band's thermally broadened Fermi surface at temperature T.\n\nbands is a vector of the . The width of the Fermi tube at each surface is pm α T. \n\n\n\n\n\n","category":"function"},{"location":"install/#Installation","page":"Installation","title":"Installation","text":"","category":"section"},{"location":"install/","page":"Installation","title":"Installation","text":"Ludwig is not yet registered on the general Julia package registry.  To run the latest version of Ludwig.jl, clone the repository. In the root folder of the project, ","category":"page"},{"location":"install/","page":"Installation","title":"Installation","text":"julia> ]\n(@v1.8) pkg> activate .\n(Ludwig) pkg> instantiate","category":"page"},{"location":"install/","page":"Installation","title":"Installation","text":"To run scripts using Ludwig, you must check out Ludwig for development in your scripting environment.","category":"page"},{"location":"install/","page":"Installation","title":"Installation","text":"julia> ]\n(@v1.8) pkg> activate PATH_TO_SCRIPT_DIR\n(SCRIPT_DIR) pkg> dev PATH_TO_LUDWIG_ROOT\n(SCRIPT_DIR) pkg> instantiate","category":"page"},{"location":"install/","page":"Installation","title":"Installation","text":"Functionality provided by Ludwig can now be accessed with using Ludwig.","category":"page"},{"location":"mesh/mesh/#Fermi-Surface-Centered-Meshes","page":"Overview","title":"Fermi Surface Centered Meshes","text":"","category":"section"},{"location":"mesh/mesh/","page":"Overview","title":"Overview","text":"At finite temperature, the particles participating in scattering with non-negligible weight belong to a narrow annulus of energies near the Fermi surface. Thus, for calculating the Boltzmann collision matrix, the sampled momenta are chosen to be uniformly distributed in angle and energy between a temperature-dependent threshhold. This follows the approach outlined in J. M. Buhmann's PhD Thesis.","category":"page"},{"location":"mesh/mesh/","page":"Overview","title":"Overview","text":"The sampled momenta lie at the orthocenter of each Patch, and information about the patch used in integration is stored in the fields of a variable of type Patch.","category":"page"},{"location":"mesh/mesh/","page":"Overview","title":"Overview","text":"Ludwig.Patch","category":"page"},{"location":"mesh/mesh/#Ludwig.Patch","page":"Overview","title":"Ludwig.Patch","text":"Representation of a patch in momentum space to be integrated over when calculating the collision integral kernel.\n\nFields\n\nmomentum: Momentum in 1st BZ scaled by 2pi  a\nenergies: Eigenvalues of Hamiltonian evaluated at momentum\nband_index: Index of the Fermi surface from which the patch was generated\nv: The group velocity at momentum taking energies[band_index] as the dispersion\ndV: Area of the patch in units of (2pia)^2\njinv: Jacobian of transformation from (k_x k_y) mapsto (varepsilon theta), the local patch coordinates\nw: The weights of the original orbital basis corresponding to the band_indexth eigenvalue\ncorners: Indices of coordinates in parent Mesh struct of corners for plotting\n\n\n\n\n\n","category":"type"},{"location":"mesh/mesh/","page":"Overview","title":"Overview","text":"Each patch belongs to a specific band, and the following method allows the energy of that patch to be accessed without storing this as a separate field.","category":"page"},{"location":"mesh/mesh/","page":"Overview","title":"Overview","text":"Ludwig.energy(p::Patch)","category":"page"},{"location":"mesh/mesh/#Ludwig.energy-Tuple{Patch}","page":"Overview","title":"Ludwig.energy","text":"energy(p::Patch)\n\nReturn the energy corresponding to the band for which p was generated.\n\n\n\n\n\n","category":"method"},{"location":"mesh/mesh/","page":"Overview","title":"Overview","text":"These patches are stored in a container type Mesh which contains additional fields for plotting.","category":"page"},{"location":"mesh/mesh/","page":"Overview","title":"Overview","text":"Ludwig.Mesh","category":"page"},{"location":"mesh/mesh/#Ludwig.Mesh","page":"Overview","title":"Ludwig.Mesh","text":"Container struct for patches over which to integrate.\n\nFields\n\npatches: Vector of patches\ncorners: Vector of points on patch corners for plotting mesh\nn_bands: Dimension of the generating Hamiltonian\n\n\n\n\n\n","category":"type"},{"location":"mesh/mesh/#References","page":"Overview","title":"References","text":"","category":"section"},{"location":"mesh/mesh/","page":"Overview","title":"Overview","text":"J. M. Buhmann, Unconventional Transport Properties of Correlated Two-Dimensional Fermi Liquids, Ph.D. thesis, Institute for Theoretical Physics ETH Zurich (2013).","category":"page"},{"location":"#Ludwig.jl-Documentation","page":"Home","title":"Ludwig.jl Documentation","text":"","category":"section"},{"location":"#Overview","page":"Home","title":"Overview","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Ludwig provides a framework for generating the linearized Boltzmann collision operator for electron-electron scattering in two-dimensional materials and materials with a pseudo-two-dimensional band structure. This package also provides utilities for calculating conductivities and viscosities of the electron fluid from the generated collision matrix. For now, only square Brillouin Zones are supported.","category":"page"},{"location":"","page":"Home","title":"Home","text":"info: Unicode\nThis package uses Unicode characters (primarily Greek letters) such as η, σ, and ε in both function names and for function arguments.  Unicode symbols can be entered in the Julia REPL by typing, e.g., \\eta followed by tab key. Read more about Unicode  symbols in the Julia Documentation.","category":"page"},{"location":"#Units","page":"Home","title":"Units","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"For all calculations, hbar = k_B = 1 For converting output back to physical units, Ludwig includes the values of some important physical constants from the 2022 CODATA Recommended Values of the Fundamental Physical Constants.","category":"page"},{"location":"","page":"Home","title":"Home","text":"G0","category":"page"},{"location":"#Ludwig.G0","page":"Home","title":"Ludwig.G0","text":"Conductance quantum in Siemens\n\n\n\n\n\n","category":"constant"},{"location":"","page":"Home","title":"Home","text":"kb","category":"page"},{"location":"#Ludwig.kb","page":"Home","title":"Ludwig.kb","text":"Boltzmann constant in eV/K\n\n\n\n\n\n","category":"constant"},{"location":"","page":"Home","title":"Home","text":"hbar","category":"page"},{"location":"#Ludwig.hbar","page":"Home","title":"Ludwig.hbar","text":"Reduced Planck's constant in eV.s\n\n\n\n\n\n","category":"constant"},{"location":"","page":"Home","title":"Home","text":"e_charge","category":"page"},{"location":"#Ludwig.e_charge","page":"Home","title":"Ludwig.e_charge","text":"Electron charge in C\n\n\n\n\n\n","category":"constant"},{"location":"","page":"Home","title":"Home","text":"danger: Energy Scale\nSince we take k_B = 1, temperatures must be expressed in the same energy scale used by the Hamiltonian.  We recommend expressing all energies in units of eV for simplicity in multiband calculations where each band may have an independent natural energy scale. This is particularly important since many function involve the ratio of the energy to temperature; e.g. f0(E, T)","category":"page"},{"location":"","page":"Home","title":"Home","text":"f0","category":"page"},{"location":"#Ludwig.f0","page":"Home","title":"Ludwig.f0","text":"f0(E, T)\n\nReturn the value of the Fermi-Dirac distribution for energy E and temperature T.\n\n    f^(0)(varepsilon) = frac11 + e^varepsilonk_B T\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"Moreover, all crystal momenta are normalized by 2pi  a_i where a_i denotes the the lattice spacing. This makes the computation of momentum integrals simplified:","category":"page"},{"location":"","page":"Home","title":"Home","text":"int fracd^2mathbfk(2pi)^2 mapsto frac1a^2 int d^2mathbfk","category":"page"},{"location":"#Other-Utilities","page":"Home","title":"Other Utilities","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Ludwig.map_to_first_bz","category":"page"},{"location":"#Ludwig.map_to_first_bz","page":"Home","title":"Ludwig.map_to_first_bz","text":"map_to_first_bz(k)\n\nMap a vector k to the d-dimensional centered unit cube where d is the dimension of k. \n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"Ludwig.get_bands","category":"page"},{"location":"#Ludwig.get_bands","page":"Home","title":"Ludwig.get_bands","text":"get_bands(H, N)\n\nReturn an interpolation of the eigenvalues of H on a square grid [-0.5, 0.5].\n\nIt is assumed that H is a function of a vector of length 2 and returns a square matrix. N is the number of points between -0.5 and 0.5 used for interpolation.\n\n\n\n\n\n","category":"function"},{"location":"#References","page":"Home","title":"References","text":"","category":"section"},{"location":"mesh/marching_squares/#Marching-Squares-Algorithm","page":"Marching Squares","title":"Marching Squares Algorithm","text":"","category":"section"},{"location":"mesh/marching_squares/","page":"Marching Squares","title":"Marching Squares","text":"Ludwig generates the energy contours for its Fermi-surface centered meshes using a simple marching squares algorithm.  The implementation is inspired by the implementation in Contour.jl, but modified so that the generated contours are angle-ordered in the first quadrant on either side of the umklapp surface. The marching squares algorithm stores the resulting contours in the convenient Isoline Bundle which stores information about the size and topology of the Fermi surface. ","category":"page"},{"location":"mesh/marching_squares/","page":"Marching Squares","title":"Marching Squares","text":"Ludwig.Isoline","category":"page"},{"location":"mesh/marching_squares/#Ludwig.Isoline","page":"Marching Squares","title":"Ludwig.Isoline","text":"Representation of a contour as an ordered set of discrete points\n\nFields\n\npoints: Vector of points in contour\nisclosed: Boolean which is true if the contour returned to the starting point when being generated\narclength: Length of the contour\n\n\n\n\n\n","category":"type"},{"location":"mesh/marching_squares/","page":"Marching Squares","title":"Marching Squares","text":"Ludwig.IsolineBundle","category":"page"},{"location":"mesh/marching_squares/#Ludwig.IsolineBundle","page":"Marching Squares","title":"Ludwig.IsolineBundle","text":"Collection of contours generated from a matrix.\n\nFields\n\nisolines: Vector of Isolines\nlevel: Constant value along contours in isolines\n\n\n\n\n\n","category":"type"},{"location":"example/#A-Basic-Example-of-Multiband-Electron-Electron-Scattering","page":"Example","title":"A Basic Example of Multiband Electron-Electron Scattering","text":"","category":"section"},{"location":"example/","page":"Example","title":"Example","text":"In this example, the function main takes a matrix-valued function H as the single-electron Hamiltonian along with the convenience function eigenvecs! which populates a preallocated matrix using an analytic expression for the eigenvectors (this improves the performance of the code by reducing allocations whien computing eigenvectors of H). The electron-electron collision matrix is created in memory and then stored in an HDF5 file. For a high resolution mesh, consider using memory mapping as the collision matrix L can become quite large to store in active memory.","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"using Ludwig\nusing HDF5\nusing Interpolations\nusing StaticArrays\nimport LinearAlgebra: eigvals\n\nfunction main(H::Function, eigenvecs!::Function, T::Real, n_ε::Int, n_θ::Int, outfile::String)\n    T = kb * T # Convert K to eV\n\n    mesh, Δε = Ludwig.multiband_mesh(H, T, n_ε, n_θ) # Default interpolation dimension and Fermi tube width\n    ℓ = length(mesh.patches)\n\n    # Initialize file\n    h5open(outfile, \"cw\") do fid\n        g = create_group(fid, \"data\")\n        write_attribute(g, \"n_e\", n_ε)\n        write_attribute(g, \"n_theta\", n_θ)\n        g[\"corners\"] = copy(transpose(reduce(hcat, mesh.corners)))\n        g[\"momenta\"] = copy(transpose(reduce(hcat, map(x -> x.momentum, mesh.patches))))\n        g[\"velocities\"] = copy(transpose(reduce(hcat, map(x -> x.v, mesh.patches))))\n        g[\"energies\"] = map(x -> x.energy, mesh.patches) \n        g[\"dVs\"] = map(x -> x.dV, mesh.patches)\n        g[\"corner_ids\"] = copy(transpose(reduce(hcat, map(x -> x.corners, mesh.patches))))\n    end \n\n    L = zeros(Float64, ℓ, ℓ) # Scattering operator\n\n    N = 1001 # Interpolation dimension for energies\n    Ludwig.electron_electron!(L, mesh.patches, Δε, T, hamiltonian, N, eigenvecs!)\n\n    # Write scattering operator out to file\n    h5open(outfile, \"cw\") do fid\n        g = fid[\"data\"]\n        g[\"L\"] = L\n    end\nend","category":"page"}]
}
