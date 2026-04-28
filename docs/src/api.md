# API Reference

## Lattices

```@autodocs
Modules = [Ludwig]
Private = false
Pages = ["lattices.jl"]
```

## Groups

```@autodocs
Modules = [Ludwig]
Private = false
Pages = ["groups.jl"]
```

## Mesh

```@autodocs
Modules = [Ludwig]
Private = false
Pages = ["fs_mesh.jl", "bands.jl", "marching_squares.jl"]
```

## Collision Operator

```@autodocs
Modules = [Ludwig]
Private = false
Pages = ["integration.jl"]
```

## Transport Properties

```@autodocs
Modules = [Ludwig]
Private = false
Pages = ["properties.jl"]
```

## Utilities

```@autodocs
Modules = [Ludwig]
Private = false
Pages = ["utilities.jl", "constants.jl"]
```

## Visualization

These functions are provided by the `LudwigMakieExt` extension and are only available when a Makie backend (e.g. `CairoMakie`) is loaded.

```@docs
plot_mesh!
plot_mesh
```
