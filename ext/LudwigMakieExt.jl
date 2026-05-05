module LudwigMakieExt

using Ludwig
import Makie

###
### plot_brillouin_zone
###

function Ludwig.plot_brillouin_zone(
    bz::Ludwig.BrillouinZone{3};
    axis=(;),
    figure=(;),
    facecolor=(:steelblue, 0.35),
    edgecolor=:black,
    edgewidth=2,
    vertexcolor=:black,
    vertexsize=10,
)
    fig = Makie.Figure(; figure...)
    ax = Makie.Axis3(fig[1, 1]; aspect=:equal, axis...)

    points = [Makie.Point3f(v) for v in bz.vertices]

    tris = NTuple{3, Int}[]
    for f in bz.facets
        for i in 2:length(f.vertices) - 1
            push!(tris, (f.vertices[1], f.vertices[i], f.vertices[i + 1]))
        end
    end
    faces = [Makie.GeometryBasics.TriangleFace(t...) for t in tris]
    msh = Makie.GeometryBasics.Mesh(points, faces)
    p = Makie.mesh!(ax, msh; color=facecolor, transparency=true)

    segs = Makie.Point3f[]
    for (i, j) in bz.ridges
        push!(segs, points[i], points[j])
    end
    Makie.linesegments!(ax, segs; color=edgecolor, linewidth=edgewidth)

    return Makie.FigureAxisPlot(fig, ax, p)
end

end # module LudwigMakieExt
