module LudwigMakieExt

using Ludwig
import Makie

###
### plot_polytope! / plot_polytope (internal): render any AbstractConvexPolytope
###

function plot_polytope!(
    ax,
    p::Ludwig.AbstractConvexPolytope{2};
    facecolor=(:steelblue, 0.35),
    edgecolor=:black,
    edgewidth=2,
)
    # Sort vertices CCW around the centroid so they form the polygon boundary.
    c = sum(p.vertices) / length(p.vertices)
    angles = [atan(v[2] - c[2], v[1] - c[1]) for v in p.vertices]
    points = [Makie.Point2f(p.vertices[i]) for i in sortperm(angles)]

    return Makie.poly!(ax, Makie.Polygon(points);
        color=facecolor,
        strokecolor=edgecolor,
        strokewidth=edgewidth,
    )
end

function plot_polytope!(
    ax,
    p::Ludwig.AbstractConvexPolytope{3};
    facecolor=(:steelblue, 0.35),
    edgecolor=:black,
    edgewidth=2,
)
    points = [Makie.Point3f(v) for v in p.vertices]

    tris = NTuple{3, Int}[]
    for f in p.facets
        for i in 2:length(f.vertices) - 1
            push!(tris, (f.vertices[1], f.vertices[i], f.vertices[i + 1]))
        end
    end
    faces = [Makie.GeometryBasics.TriangleFace(t...) for t in tris]
    msh = Makie.GeometryBasics.Mesh(points, faces)
    plt = Makie.mesh!(ax, msh; color=facecolor, transparency=true)

    segs = Makie.Point3f[]
    for (i, j) in p.ridges
        push!(segs, points[i], points[j])
    end
    Makie.linesegments!(ax, segs; color=edgecolor, linewidth=edgewidth)

    return plt
end

function plot_polytope(
    p::Ludwig.AbstractConvexPolytope{2};
    axis=(;),
    figure=(;),
    kwargs...,
)
    fig = Makie.Figure(; figure...)
    ax = Makie.Axis(fig[1, 1];
        aspect=Makie.DataAspect(),
        xgridvisible=false,
        ygridvisible=false,
        axis...,
    )
    plt = plot_polytope!(ax, p; kwargs...)
    return Makie.FigureAxisPlot(fig, ax, plt)
end

function plot_polytope(
    p::Ludwig.AbstractConvexPolytope{3};
    axis=(;),
    figure=(;),
    kwargs...,
)
    fig = Makie.Figure(; figure...)
    ax = Makie.Axis3(fig[1, 1]; aspect=:equal, axis...)
    plt = plot_polytope!(ax, p; kwargs...)
    return Makie.FigureAxisPlot(fig, ax, plt)
end

###
### Public entry points
###

Ludwig.plot_brillouin_zone(bz::Ludwig.BrillouinZone; kwargs...) =
    plot_polytope(bz; kwargs...)

Ludwig.plot_brillouin_zone!(ax, bz::Ludwig.BrillouinZone; kwargs...) =
    plot_polytope!(ax, bz; kwargs...)

Ludwig.plot_irreducible_brillouin_zone(
    ibz::Ludwig.IrreducibleBrillouinZone; kwargs...
) = plot_polytope(ibz; kwargs...)

Ludwig.plot_irreducible_brillouin_zone!(
    ax, ibz::Ludwig.IrreducibleBrillouinZone; kwargs...
) = plot_polytope!(ax, ibz; kwargs...)

end # module LudwigMakieExt
