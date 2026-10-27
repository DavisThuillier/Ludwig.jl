using Pkg
Pkg.activate(@__DIR__)

using CairoMakie
using StaticArrays
using DelimitedFiles

include(joinpath(@__DIR__, "local_settings.jl"))
const SETTINGS = LOCAL_SETTINGS.plot_2d_quadrature_rule

include(joinpath(@__DIR__, "data_io.jl"))

function main(path)
    rule = load_quadrature_rule(path)
    nodes, weights = rule.nodes, rule.weights
    @show length(nodes)

    fig = Figure(size = (500, 500))
    ax = Axis(fig[1, 1];
        aspect = DataAspect(),
        xlabel = L"k_x",
        ylabel = L"k_y",
        xgridvisible = false,
        ygridvisible = false
    )

    SETTINGS.xlims === nothing || xlims!(ax, SETTINGS.xlims...)
    SETTINGS.ylims === nothing || ylims!(ax, SETTINGS.ylims...)

    node_plot = scatter!(ax, nodes; color = weights, markersize = 3)#, colorscale = log10)
    Colorbar(fig[1, 2], node_plot; label = L"w")

    display(fig)
end

main(joinpath(@__DIR__, "..", "meshes", SETTINGS.quad_file))
