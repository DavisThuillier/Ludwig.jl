using Ludwig
using LinearAlgebra

# Run validate_mesh.jl --plot to visualize the mesh with CairoMakie
if "--plot" in ARGS
    using CairoMakie, GeometryBasics, LaTeXStrings
end

function (@main)(args)
    ## Model ##############################################################

    # Square lattice with unit lattice constant
    l = Lattice([1.0 0.0; 0.0 1.0])

    # NNTB dispersion (energies in eV, momenta in units of 2π/a)
    t = 1.0   # eV
    μ = -1.0  # eV
    ε(k) = -2t * (cos(2π * k[1]) + cos(2π * k[2])) - μ

    # Temperature (in eV, same units as the Hamiltonian)
    T = 0.025  # ≈ 290 K

    ## Mesh generation ####################################################

    n_levels = 7   # number of energy contour boundaries (n_levels - 1 patches in energy)
    n_cuts   = 12  # number of angular cuts along the Fermi surface (n_cuts - 1 patches per sector)

    mesh = ibz_mesh(l, [ε], T, n_levels, n_cuts)
    grid = mesh.patches
    N    = length(grid)

    println("Number of IBZ patches: ", N)

    ## Patch inspection ###################################################

    p = grid[1]
    println("\nFirst patch:")
    println("  Energy:      ", energy(p), " eV")
    println("  Momentum:    ", momentum(p))
    println("  Velocity:    ", velocity(p), " eV·a / 2π")
    println("  Patch area:  ", p.dV)
    println("  Energy width:", p.de, " eV")

    ## Validation #########################################################

    α = 6.0
    E = energy.(grid)

    # 1. Energy bounds: all patches must lie within ±αT of the Fermi energy
    @assert all(abs.(E) .< α * T) "Some patches lie outside the Fermi tube!"
    println("\nEnergy range: [", minimum(E), ", ", maximum(E), "] eV  ✓")

    # 2. Nonzero velocities: every patch must have a nonzero group velocity
    speeds = norm.(velocity.(grid))
    @assert all(speeds .> 0) "Zero-velocity patch found!"
    println("Speed range:  [", minimum(speeds), ", ", maximum(speeds), "] eV·a / 2π  ✓")

    # 3. Patch area coverage: compare against a fine-grid estimate of the Fermi tube area.
    #    For the square lattice the IBZ is the triangle kx ∈ [0, 0.5], ky ∈ [0, kx].
    #    Count grid cells inside both the IBZ and the energy window |ε(k)| < αT.
    total_dV = sum(p.dV for p in grid)
    Ng  = 10000
    dk  = 0.5 / Ng
    area_est = 0.0
    for i in 0:(Ng - 1), j in 0:(Ng - 1)
        kx = (i + 0.5) * dk
        ky = (j + 0.5) * dk
        ky > kx && continue
        abs(ε([kx, ky])) < α * T && (area_est += dk^2)
    end
    @assert isapprox(total_dV, area_est; rtol = 0.05) "Patch area deviates >5% from grid estimate! (sum dV = $total_dV, estimate = $area_est)"
    println("Σ dV: ", total_dV, ", tube area estimate: ", area_est, "  ✓")

    ## Full BZ mesh #######################################################

    bz = bz_mesh(l, [ε], T, n_levels, n_cuts)
    println("\nIBZ patches: ", length(mesh.patches))
    println("BZ patches:  ", length(bz.patches), "  (should be 8×)")
    @assert length(bz.patches) == 8 * length(mesh.patches) "BZ patch count is not 8× the IBZ patch count!"

    ## Visualization (requires CairoMakie and GeometryBasics) #############

    if "--plot" in args
        polys = [Polygon([Point2f(mesh.corners[j]) for j in mesh.corner_inds[i]]) for i in 1:N]

        fig = Figure()
        ax  = Axis(fig[1, 1]; 
            aspect = DataAspect(), 
            xlabel = L"k_x (2π / a_x)", 
            ylabel = L"k_y (2π / a_y)",
            xgridvisible = false,
            ygridvisible = false,
            title = "IBZ Fermi surface mesh (NNTB, μ = $μ eV, T = $T eV)"
        )
        pl  = poly!(ax, polys; color = E, colormap = :viridis)
        ibz_verts = get_ibz(l)
        lines!(ax, Point2f.(vcat(ibz_verts, [ibz_verts[1]])); color = :black, linewidth = 1.5)
        Colorbar(fig[1, 2], pl; label = "Energy (eV)")
        display(fig)
        println("\nPlot displayed. Close the window to exit.")
    end
end
