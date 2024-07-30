# Fermi Surface Centered Meshes

At finite temperature, the particles participating in scattering with non-negligible weight belong to a narrow annulus of energies near the Fermi surface. Thus, for calculating the Boltzmann collision matrix, the sampled momenta are chosen to be uniformly distributed in angle and energy between a temperature-dependent threshhold. This follows the approach outlined in [J. M. Buhmann's PhD Thesis](https://www.research-collection.ethz.ch/handle/20.500.11850/153996).

The sampled momenta lie at the orthocenter of each Patch, and information about the patch used in integration is stored in the fields of a variable of type `Patch`.
```@docs
Ludwig.Patch
```
These patches are stored in a container type `Mesh` which contains additional fields for plotting.
```@docs
Ludwig.Mesh
```



## References
J. M. Buhmann, Unconventional Transport Properties of Correlated Two-Dimensional Fermi Liquids, Ph.D. thesis, Institute
for Theoretical Physics ETH Zurich (2013).