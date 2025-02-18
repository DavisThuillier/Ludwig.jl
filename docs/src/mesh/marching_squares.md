# Marching Squares Algorithm

Ludwig generates the energy contours for its Fermi-surface centered meshes using a simple marching squares algorithm. 
This implementation is inspired by [Contour.jl](https://juliageometry.github.io/Contour.jl/stable/index.html), but has been modified to generate contours bounded by a convex polygon. This is useful for generating the contours only within the (irreducible) Brillouin zone to create meshes that fully preserve the symmetry of the lattice.

Resulting contours are returned in the convenient `Isoline Bundle` which stores information about the size and topology of the Fermi surface. 

```@docs
Ludwig.Isoline
```

```@docs
Ludwig.IsolineBundle
```