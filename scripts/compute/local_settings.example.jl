###
### Local script settings — documented template
###
### Copy this file to `local_settings.jl` beside it and fill in your own values:
###
###     cp local_settings.example.jl local_settings.jl
###
### `local_settings.jl` is gitignored, so each developer keeps their own copy and nobody's machine
### paths or dataset names reach the repository. This template is tracked; when a script gains or
### loses a setting, update it here too.
###
### Every script under scripts/compute_scripts that reads `LOCAL_SETTINGS` takes one entry, keyed by
### its own filename without the extension, e.g. diagnose_collision_symmetry.jl reads
### `LOCAL_SETTINGS.diagnose_collision_symmetry`. A missing key is a load-time error in that script,
### so keep an entry for each script you run.
###
### Only frequently-tweaked runtime knobs belong here — which dataset to load, where to write.
### Algorithmic parameters (refinement depth, mesh and level-set bounds, quadrature sizes) stay
### inline in the scripts; they are logic, not configuration.
###
### Conventions used below:
###   collision_file   a file in scripts/data/ written by the compute scripts
###

const LOCAL_SETTINGS = (
    # diagnose_collision_symmetry.jl — asymmetry of the balanced operator Diagonal(D)*L_full,
    # measured against the depth of the negative eigenvalues of the symmetrized operator.
    diagnose_collision_symmetry = (
        collision_file = "your_collision_file.h5",
    ),
)
