###
### Plot the spectrum of the diagonalized electron-electron collision operator
###
### Reads the `eigenvalues` dataset written by diagonalize_collision_operator.jl and draws the
### sorted eigenvalues λᵢ against their index on a single linear panel.
###

using Pkg
Pkg.activate(@__DIR__)

using CairoMakie

include(joinpath(@__DIR__, "local_settings.jl"))
const SETTINGS = LOCAL_SETTINGS.plot_collision_spectrum

include(joinpath(@__DIR__, "data_io.jl"))

function main()
    collision_files = map(file -> joinpath(@__DIR__, "..", "data", file), SETTINGS.collision_files)

    spectra = map(collision_files) do path
        data = load_spectrum(path)
        (; λ = sort(data.eigenvalues), data.provenance.model_name, data.provenance.T_kelvin, path)
    end

    idx = 2
    spectra[idx] =(; λ = spectra[idx].λ, spectra[idx].model_name, spectra[idx].T_kelvin, spectra[idx].path)

    # Series sharing a model_name/T_kelvin are disambiguated by appending the file basename.
    provenance = [(spectrum.model_name, spectrum.T_kelvin) for spectrum ∈ spectra]
    function series_label(spectrum)
        label = "$(spectrum.model_name), T = $(spectrum.T_kelvin) K"
        count(==((spectrum.model_name, spectrum.T_kelvin)), provenance) > 1 ?
            "$label ($(basename(spectrum.path)))" : label
    end

    fig = Figure(size = (700, 450))
    ax = Axis(fig[1, 1];
        xlabel = L"i",
        ylabel = L"\lambda_i",
        xgridvisible = false,
        # yscale = log10
    )
    SETTINGS.xlims === nothing || xlims!(ax, SETTINGS.xlims...)
    SETTINGS.ylims === nothing || ylims!(ax, SETTINGS.ylims...)

    for spectrum ∈ spectra
        scatter!(ax, 2:length(spectrum.λ), spectrum.λ[2:end];
            markersize = 5, label = series_label(spectrum))
    end

    axislegend(ax; position = :rb)

    output_path = joinpath(@__DIR__, "..", "plots", SETTINGS.output)
    save(output_path, fig)
    @info "wrote $output_path"

    display(fig)

    return nothing
end

main()
