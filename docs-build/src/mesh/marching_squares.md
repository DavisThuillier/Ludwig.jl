# Marching Squares Algorithm

Ludwig generates the energy contours for its Fermi-surface centered meshes using a simple marching squares algorithm. 
The implementation is inspired by the implementation in [Contour.jl](https://juliageometry.github.io/Contour.jl/stable/index.html), but modified so that the generated contours are angle-ordered in the first quadrant on either side of the umklapp surface. The marching squares algorithm stores the resulting contours in the convenient `Isoline Bundle` which stores information about the size and topology of the Fermi surface. 

```@docs
Ludwig.Isoline
```

```@docs
Ludwig.IsolineBundle
```