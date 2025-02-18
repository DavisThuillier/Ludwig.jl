# Ludwig.jl Documentation

## Overview
Ludwig provides a framework for generating the linearized Boltzmann collision operator for electron-electron scattering in two-dimensional materials and materials with a quasi-2D band structure. This package also provides utilities for calculating transport properties such as conductivity and viscosity from the generated collision matrix.

!!! info "Unicode"
    This package uses Unicode characters (primarily Greek letters) such as `η`, `σ`, and `ε` in both function names and for function arguments. 
    Unicode symbols can be entered in the Julia REPL by typing, e.g., `\eta` followed by `tab` key. Read more about Unicode 
    symbols in the [Julia Documentation](https://docs.julialang.org/en/v1/manual/unicode-input/).

## Units
For all calculations, ``\hbar = k_B = 1.`` For converting output back to physical units, Ludwig includes the values of some important physical constants from the [2022 CODATA Recommended Values of the Fundamental Physical Constants](https://physics.nist.gov/constants).
```@docs
G0
```
```@docs
kb
```
```@docs
hbar
```
```@docs
e_charge
```

!!! danger "Energy Scale" 
    Since we take ``k_B = 1``, temperatures must be expressed in the same energy scale used by the Hamiltonian. 
    We recommend expressing all energies in units of eV for simplicity in multiband calculations where each band may have an independent natural energy scale. This is particularly important since many function involve the ratio of the energy to temperature; e.g. `f0(E, T)`
    
```@docs
f0
```

Moreover, all momentum integrals are normalized by the volume of the Brillouin zone. This simplifies calculations, but again requires appropriate dimension to be restored later:
```math
\int \frac{d^2\mathbf{k}}{(2\pi)^2} \to \frac{1}{v_\text{cell}} \int d^2\mathbf{k}
```

## References

