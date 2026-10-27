###
### Local script settings — documented template
###
### Copy this file to `local_settings.jl` beside it and fill in your own values:
###
###     cp local_settings.example.jl local_settings.jl
###
### `local_settings.jl` is gitignored, so each developer keeps their own copy and nobody's machine
### paths, dataset names, or axis limits reach the repository. This template is tracked; when a script
### gains or loses a setting, update it here too.
###
### Every script under scripts/plotting_scripts reads one entry of `LOCAL_SETTINGS`, keyed by its own
### filename without the extension, e.g. plot_collision_row.jl reads `LOCAL_SETTINGS.plot_collision_row`.
### A missing key is a load-time error in that script, so keep an entry for each script you run.
###
### Only frequently-tweaked runtime and cosmetic knobs belong here — which dataset to load, where to
### write, what part of k-space to look at. Algorithmic parameters (refinement depth, mesh and level-set
### bounds, quadrature sizes) stay inline in the scripts; they are logic, not configuration.
###
### Conventions used below:
###   material_file    a file in scripts/materials/ supplying dispersion/velocity/primitives/generators
###   collision_file   a file in scripts/data/ written by the compute scripts
###   quad_file        a .quad rule in scripts/meshes/
###   output           a filename written into scripts/plots/
###   xlims/ylims/zlims   Cartesian k limits, in units of inverse lattice spacing (a ≡ 1)
###

const LOCAL_SETTINGS = (
    # plot_2d_quadrature_rule.jl — scatter of a 2D .quad rule coloured by weight.
    plot_2d_quadrature_rule = (
        quad_file = "your_rule.quad",
        xlims = (0.0, 1.0),
        ylims = (0.0, 1.0),
    ),

    # plot_3d_quadrature_rule.jl — scatter of a 3D .quad rule coloured by weight.
    # `nothing` on any axis leaves that limit to Makie's autoscaling.
    plot_3d_quadrature_rule = (
        quad_file = "your_3d_rule.quad",
        xlims = nothing,
        ylims = nothing,
        zlims = nothing,
    ),

    # fermi_surface_2d.jl — Fermi contour and refinement study. `T` is a temperature in eV.
    fermi_surface_2d = (
        T = 0.1,
        xlims = (0.0, 1.0),
        ylims = (0.0, 1.0),
    ),

    # fermi_surface_3d.jl — tight-binding Fermi surface, saved to scripts/plots/.
    fermi_surface_3d = (
        material_file = "your_material.jl",
        output = "fermi_surface_3d.png",
    ),

    # plot_collision_row.jl — one row L_{i,j} of the collision operator over the Fermi annulus.
    # `row_indices` selects which rows to step through; `nothing` limits leave the axes autoscaled.
    plot_collision_row = (
        material_file = "your_material.jl",
        collision_file = "your_collision_file.h5",
        row_indices = 1:10,
        xlims = nothing,
        ylims = nothing,
    ),

    # plot_eigenvector_modes_folded.jl — the `n_modes` slowest eigenvector modes, folded into whole
    # Fermi pockets, one panel each.
    # Panel windows are derived from the folded nodes, so there are no xlims/ylims here.
    #   window_pad  panel half-width as a multiple of the pocket radius; > 1 leaves a margin
    #   show_seams  overlay the translated zone boundaries that meet at each pocket centre, marking
    #               the seams between the corner wedges the pocket is stitched from
    plot_eigenvector_modes_folded = (
        material_file = "your_material.jl",
        collision_file = "your_collision_file.h5",
        n_modes = 12,
        window_pad = 1.15,
        show_seams = true,
    ),

    # plot_collision_spectrum.jl — sorted eigenvalue spectra overlaid across datasets.
    # `collision_files` is a list; one curve is drawn per file, so keep them comparable.
    plot_collision_spectrum = (
        collision_files = [
            "your_collision_file.h5",
            "your_other_collision_file.h5",
        ],
        output = "collision_spectrum_overlay.png",
        xlims = (0, 100),
        ylims = (0, 1e-7),
    ),

    # irrep_test.jl — modes of one isotypic block of the collision matrix in the class-sum eigenbasis,
    # where the isotypic components of the permutation representation appear as diagonal blocks. The
    # rule must be the one the collision file was built on. `window_pad` and `show_seams` control the
    # folded-pocket eigenvector panels, as in plot_eigenvector_modes_folded.jl.
    irrep_test = (
        material_file = "your_material.jl",
        quad_file = "your_rule.quad",
        collision_file = "your_collision_file.h5",
        window_pad = 1.15,
        show_seams = true,
    ),
)
