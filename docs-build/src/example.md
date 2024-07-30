# A Basic Example of Multiband Electron-Electron Scattering

In this example, the function `main` takes a matrix-valued function `H` as the single-electron Hamiltonian along with the convenience function `eigenvecs!` which populates a preallocated matrix using an analytic expression for the eigenvectors (this improves the performance of the code by reducing allocations whien computing eigenvectors of `H`). The electron-electron collision matrix is created in memory and then stored in an HDF5 file. For a high resolution mesh, consider using memory mapping as the collision matrix `L` can become quite large to store in active memory.

```
using Ludwig
using HDF5
using Interpolations
using StaticArrays
import LinearAlgebra: eigvals

function main(H::Function, eigenvecs!::Function, T::Real, n_ε::Int, n_θ::Int, outfile::String)
    T = kb * T # Convert K to eV

    mesh, Δε = Ludwig.multiband_mesh(H, T, n_ε, n_θ) # Default interpolation dimension and Fermi tube width
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

    L = zeros(Float64, ℓ, ℓ) # Scattering operator

    N = 1001 # Interpolation dimension for energies
    Ludwig.electron_electron!(L, mesh.patches, Δε, T, hamiltonian, N, eigenvecs!)

    # Write scattering operator out to file
    h5open(outfile, "cw") do fid
        g = fid["data"]
        g["L"] = L
    end
end
```