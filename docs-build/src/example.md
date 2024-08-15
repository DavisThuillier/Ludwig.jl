# A Basic Example of Multiband Electron-Electron Scattering

In this example, the function `main` takes a vector of functions `bands` as the eigenvalues of the single-electron Hamiltonian along with the convenience function `orbital_weights` which returns a matrix using an analytic expression for the eigenvectors. The electron-electron collision matrix is created in memory and then stored in an HDF5 file. For a high resolution mesh, consider using memory mapping as the collision matrix `L` can become quite large to store in active memory.

```
using Ludwig
using HDF5
using Interpolations
using StaticArrays
using LinearAlgebra

function main(bands, orbital_weights, T, n_ε, n_θ, outfile)
    T = kb * T # Convert K to eV

    mesh = Ludwig.multiband_mesh(bands, orbital_weights, T, n_ε, n_θ)
    ℓ = length(mesh.patches)

    # Initialize file
    h5open(outfile, "cw") do fid
        g = create_group(fid, "data")
        write_attribute(g, "n_e", n_ε)
        write_attribute(g, "n_theta", n_θ)
        g["corners"] = copy(transpose(reduce(hcat, mesh.corners)))
        g["momenta"] = copy(transpose(reduce(hcat, map(x -> x.momentum, mesh.patches))))
        g["velocities"] = copy(transpose(reduce(hcat, map(x -> x.v, mesh.patches))))
        g["energies"] = map(x -> x.energy, mesh.patches) 
        g["dVs"] = map(x -> x.dV, mesh.patches)
        g["corner_ids"] = copy(transpose(reduce(hcat, map(x -> x.corners, mesh.patches))))
    end 

    N = 1001
    x = LinRange(-0.5, 0.5, N)
    E = Array{Float64}(undef, N, N)
    itps = Vector{ScaledInterpolation}(undef, length(bands))
    for μ in eachindex(bands)
        for i in 1:N, j in 1:N
            E[i, j] = bands[μ]([x[i], x[j]]) # Get eigenvalues (bands) of each k-point
        end

        itp = interpolate(E, BSpline(Cubic(Line(OnGrid()))))
        itps[μ] = scale(itp, x, x)
    end

    L = zeros(Float64, ℓ, ℓ) # Scattering operator
    f0s = map(x -> f0(x.energy, T), mesh.patches) # Fermi-Dirac Grid

    for i in 1:ℓ
        for j in 1:ℓ
            L[i,j] = Ludwig.electron_electron(mesh.patches, f0s, i, j, itps, T, vertex_pp, vertex_pk)
        end
    end

    # Write scattering operator out to file
    h5open(outfile, "cw") do fid
        g = fid["data"]
        g["L"] = L
    end
end
```

