# Ludwig.jl

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://davisthuillier.github.io/Ludwig.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://davisthuillier.github.io/Ludwig.jl/dev)

Ludwig.jl computes the linearized Boltzmann collision operator electron-electron scattering. From the resulting collision matrix it derives transport coefficients — electrical and thermal conductivity, thermopower, viscosity, and effective scattering lifetimes — using a linear-solver back end of the user's choice.

The package uses $\hbar = k_B = 1$ internally; the convention throughout the public API is to express energies and temperatures in eV and momenta in inverse lattice constants.

## Installation

Ludwig.jl is registered in the Julia General registry:

```julia
julia> ]
pkg> add Ludwig
```

For the development version:

```julia
pkg> add https://github.com/DavisThuillier/Ludwig.jl
```

## Citing Ludwig.jl

If you use Ludwig.jl in research, please cite it.

```bibtex
@software{LudwigJl,
  author  = {Thuillier, Davis and Scaffidi, Thomas},
  title   = {{Ludwig.jl}},
  url     = {https://github.com/DavisThuillier/Ludwig.jl},
  version = {0.4.0},
  year    = {2026},
}
```

## License

Ludwig.jl is released under the [MIT license](LICENSE.md).

## Authors

[Davis Thuillier](https://github.com/DavisThuillier) and Thomas Scaffidi.
