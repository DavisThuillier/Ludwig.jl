module LudwigMakieExt

using Ludwig
import Makie
using LaTeXStrings

###
### plot_mesh!
###

function Ludwig.plot_mesh!(
    ax,
    mesh::Ludwig.Mesh;
    color       = nothing,
    colormap    = :viridis,
    colorrange  = nothing,
    strokecolor = :black,
    strokewidth = 0.3,
)
    n        = length(mesh.patches)
    energies = Ludwig.energy.(mesh.patches)
    polys    = [
        Makie.Polygon([Makie.Point2f(mesh.corners[j]) for j in mesh.corner_indices[i]])
        for i in 1:n
    ]

    c  = isnothing(color) ? energies : color
    cr = if isnothing(colorrange)
        emax = maximum(abs, energies)
        (-emax, emax)
    else
        colorrange
    end

    return Makie.poly!(ax, polys;
        color       = c,
        colormap    = colormap,
        colorrange  = cr,
        strokecolor = strokecolor,
        strokewidth = strokewidth,
    )
end

###
### plot_mesh
###

function Ludwig.plot_mesh(
    mesh::Ludwig.Mesh;
    axis   = (;),
    figure = (;),
    kwargs...,
)
    fig = Makie.Figure(; figure...)
    ax  = Makie.Axis(fig[1, 1];
        aspect       = Makie.DataAspect(),
        xgridvisible = false,
        ygridvisible = false,
        xlabel       = L"k_x",
        ylabel       = L"k_y",
        axis...,
    )
    p = Ludwig.plot_mesh!(ax, mesh; kwargs...)
    return Makie.FigureAxisPlot(fig, ax, p)
end

end # module LudwigMakieExt
