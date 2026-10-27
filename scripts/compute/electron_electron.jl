###
### Distributed electron-electron collision matrix fill (MPI + parallel HDF5)
###
### Reads a precomputed quadrature rule (nodes, weights, N_ibz) from disk, then fills the collision
### matrix in parallel: each MPI rank owns a contiguous block of IBZ columns, threads over the rows
### within that block, and writes its block straight into a shared HDF5 dataset -- no gather, no
### transpose.
###
### Point-group invariance,  L(g·kᵢ, g·kⱼ) = L(kᵢ, kⱼ)  for every operation g, means only the columns
### j ∈ 1:N_ibz need to be computed: every other column is one of these permuted by a group operation,
### which `full_collision_matrix` applies at analysis time. Detailed balance plays no part in that
### reconstruction. With i ∈ 1:N as the row and j ∈ 1:N_ibz as the column, column-major storage keeps
### each IBZ column contiguous, so it maps onto an HDF5 hyperslab directly.
###
### Each entry carries the column weight wⱼ and nothing else -- the row weight wᵢ enters downstream,
### through the detailed-balance factors Dᵢ. The unfolding is weight-correct because wⱼ is constant
### along each point-group orbit, so a permuted column carries the weight its new position needs.
###
### The temperature is read from the quadrature-rule header, so it is not passed on the command line.
###
### Launch (one rank per node, threads = cores/node):
###     srun -n $SLURM_NNODES julia --threads=$SLURM_CPUS_PER_TASK \
###         electron_electron.jl <quadrature_file> <material_file>
###

using Pkg
Pkg.activate(@__DIR__)

using ArgParse
using MPI
using HDF5
using DelimitedFiles
using ProgressMeter
using Ludwig

function argument_handling(ARGS)
    settings = ArgParseSettings(description = "Fill the electron-electron collision matrix from a quadrature rule (MPI + HDF5).")

    @add_arg_table! settings begin
        "--initial_depth"
            help = "Coarse seed depth: level-set seed grid has 2^initial_depth + 1 points per axis"
            arg_type = Int
            default = 7
        "--refinement_depth"
            help = "Total adaptive-subdivision depth; finest cell spans 1 / 2^refinement_depth"
            arg_type = Int
            default = 20
        "--outdir", "-o"
            help = "Output collision-matrix HDF5 file"
            arg_type = String
            default = ""
        "--mesh_file"
            help = "Quadrature-rule file produced by generate_quadrature_rule.jl"
            arg_type = String
            required = true
        "material_file"
            help = "File defining dispersion, velocity, point-group generators, and lattice vectors"
            arg_type = String
            required = true
    end

    return parse_args(ARGS, settings)
end

"""
    load_quadrature_rule(path) -> (nodes, weights, N_ibz, T_kelvin, energy_window)

Read the quadrature rule written by `generate_quadrature_rule.jl`: `N_ibz`, the build
temperature `T_kelvin` (Kelvin), and the thermal `energy_window` (eV) from the header, then
the `k…, w` data rows as `nodes::Vector{SVector{D,Float64}}` and `weights::Vector{Float64}`.
"""
function load_quadrature_rule(path)
    N_ibz = 0
    T_kelvin = NaN
    energy_window = NaN
    open(path) do io
        for line ∈ eachline(io)
            startswith(line, '#') || break
            ibz_match = match(r"N_ibz\s*=\s*(\d+)", line)
            ibz_match === nothing || (N_ibz = parse(Int, ibz_match.captures[1]))
            temperature_match = match(r"^#\s*T\b[^=]*=\s*([-+\d.eE]+)", line)
            temperature_match === nothing || (T_kelvin = parse(Float64, temperature_match.captures[1]))
            window_match = match(r"energy_window\b[^=]*=\s*([-+\d.eE]+)", line)
            window_match === nothing || (energy_window = parse(Float64, window_match.captures[1]))
        end
    end
    isnan(T_kelvin) && error("no temperature (`# T [K] = …`) found in the header of $path")
    isnan(energy_window) && error("no energy window (`# energy_window [eV] = …`) found in the header of $path")

    data = readdlm(path, ','; comments = true, comment_char = '#')
    D = size(data, 2) - 1
    nodes   = [SVector{D,Float64}(data[i, 1:D]) for i ∈ axes(data, 1)]
    weights = Vector{Float64}(data[:, end])

    return nodes, weights, N_ibz, T_kelvin, energy_window
end

"""
    write_metadata!(file, primitives, generators, T_kelvin, energy_window, N_ibz, N, parsed, initial_depth, refinement_depth)

Write the lattice, point-group, and provenance metadata that makes the collision file self-contained:
`lattice/primitives`, the stacked Cartesian `lattice/generators`, and a `provenance` group whose
attributes carry the build temperature, thermal energy window (eV), node counts, model name, and
marching depths.
"""
function write_metadata!(file, primitives, generators, T_kelvin, energy_window, N_ibz, N, parsed, initial_depth, refinement_depth)
    file["lattice/primitives"] = Matrix{Float64}(primitives)
    file["lattice/generators"] = cat([Matrix{Float64}(g) for g in generators]..., dims = 3)

    provenance = create_group(file, "provenance")
    attributes(provenance)["T_kelvin"]         = T_kelvin
    attributes(provenance)["energy_window"]    = energy_window
    attributes(provenance)["N_ibz"]            = N_ibz
    attributes(provenance)["N"]                = N
    attributes(provenance)["model_name"]       = @isdefined(MATERIAL_NAME) ? string(MATERIAL_NAME) : basename(parsed["material_file"])
    attributes(provenance)["initial_depth"]    = initial_depth
    attributes(provenance)["refinement_depth"] = refinement_depth
    return nothing
end

function main(parsed)
    MPI.Init()
    comm   = MPI.COMM_WORLD
    rank   = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    rank == 0 && (@info "starting electron-electron fill: nranks=$nranks"; flush(stderr))

    ###
    ### Numerical parameters
    ###
    initial_depth    = parsed["initial_depth"]
    refinement_depth = parsed["refinement_depth"]
    quadrature_file = parsed["mesh_file"]
    output_dir     = parsed["outdir"]

    stem = split(basename(quadrature_file), ".")[1]
    output_file = joinpath(output_dir, stem * "_$(refinement_depth).h5")

    ###
    ### Load the quadrature rule on rank 0 and broadcast it to the others. The temperature comes from
    ### the rule's header (Kelvin) and is converted to energy units via kb, matching how the rule itself
    ### was built.
    ###
    nodes, weights, N_ibz, T_kelvin, energy_window = if rank == 0
        load_quadrature_rule(quadrature_file)
    else
        (nothing, nothing, 0, NaN, NaN)
    end
    nodes         = MPI.bcast(nodes,         comm; root = 0)
    weights       = MPI.bcast(weights,       comm; root = 0)
    N_ibz         = MPI.bcast(N_ibz,         comm; root = 0)
    T_kelvin      = MPI.bcast(T_kelvin,      comm; root = 0)
    energy_window = MPI.bcast(energy_window, comm; root = 0)
    N        = length(nodes)   # nodes is already broadcast, so its length is valid on every rank
    T        = kb * T_kelvin   # energy units (eV)

    nranks ≤ N_ibz || error("more ranks ($nranks) than IBZ columns ($N_ibz); reduce the rank count.")

    rank == 0 && (@info "quadrature rule loaded: N=$N, N_ibz=$N_ibz; starting fill"; flush(stderr))

    ###
    ### Reconstruct the crystal-dependent maps locally (cheap, deterministic, never serialized)
    ###
    point_group = PointGroup(generators, primitives)
    crystal     = Crystal(primitives, point_group)

    ###
    ### This rank's contiguous block of IBZ columns  j ∈ column_lo:column_hi
    ###
    column_lo = ( rank      * N_ibz) ÷ nranks + 1
    column_hi = ((rank + 1) * N_ibz) ÷ nranks
    n_columns = column_hi - column_lo + 1

    # block[i, local_column] is full-BZ row i ∈ 1:N of IBZ column j = column_lo + local_column - 1.
    # Threading over the flat (row, local-column) index keeps each integral serial (threaded = false) so
    # the work units are the matrix entries themselves -- large and roughly uniform, with :dynamic
    # absorbing per-entry variance.
    block = zeros(Float64, N, n_columns)
    # rank 0 always shows a progress bar for its own column block; other ranks stay silent to avoid
    # interleaved output -- rank 0's progress is a reasonable proxy for the whole job since columns are
    # split evenly across ranks.
    progress = rank == 0 ? Progress(N * n_columns; desc = "filling L: ") : nothing
    Threads.@threads :dynamic for n in 0:(N * n_columns - 1)
        i = n % N + 1
        local_column = n ÷ N + 1
        j = column_lo + local_column - 1
        if i != j
            ki = nodes[i]
            kj = nodes[j]
            Lij = electron_electron(ki, kj, weights[j], crystal, dispersion, velocity, T, Weff_squared; threaded = false, initial_depth, refinement_depth)
            block[i, local_column] = Lij
        end
        # i == j: coincident-node self-term left at the zero-initialised diagonal value
        progress !== nothing && next!(progress)
    end

    rank == 0 && (@info "fill complete; writing collision matrix to $output_file"; flush(stderr))

    ###
    ### Auxiliary datasets that make the collision file self-contained for downstream analysis. Identical
    ### on every rank, so only rank 0 writes them.
    ###
    energies = [dispersion(nodes[i]) for i in 1:N]
    D        = detailed_balance_factors(weights, energies, T)

    ###
    ### Write each rank's column block. A lone rank (e.g. a local run) writes serially and needs only
    ### the default serial HDF5; multiple ranks write collectively into the shared dataset, which needs
    ### a parallel (MPI) HDF5 build.
    ###
    output_dir = dirname(output_file)
    rank == 0 && !isempty(output_dir) && mkpath(output_dir)
    MPI.Barrier(comm)
    if nranks == 1
        h5open(output_file, "w") do file
            file["L"] = block                      # one rank holds the full N × N_ibz matrix
            file["nodes"]    = reduce(hcat, nodes)
            file["weights"]  = weights
            file["energies"] = energies
            file["D"]        = D
            write_metadata!(file, primitives, generators, T_kelvin, energy_window, N_ibz, N,
                            PARSED, initial_depth, refinement_depth)
        end
    else
        HDF5.has_parallel() || error("multi-rank fill needs a parallel (MPI) HDF5 build " *
                                     "(HDF5.has_parallel() == true); configure HDF5.API.set_libraries! " *
                                     "or run with a single rank.")
        h5open(output_file, "w", comm, MPI.Info()) do file
            dataset = create_dataset(file, "L", datatype(Float64), dataspace((N, N_ibz));
                                     chunk = (N, 1), dxpl_mpio = :collective)
            dataset[:, column_lo:column_hi] = block
        end
        if rank == 0
            h5open(output_file, "r+") do file
                file["nodes"]    = reduce(hcat, nodes)
                file["weights"]  = weights
                file["energies"] = energies
                file["D"]        = D
                write_metadata!(file, primitives, generators, T_kelvin, energy_window, N_ibz, N,
                                PARSED, initial_depth, refinement_depth)
            end
        end
        MPI.Barrier(comm)
    end

    rank == 0 && (@info "wrote $(N)×$(N_ibz) collision matrix to $output_file"; flush(stderr))

    MPI.Finalize()
    println(output_file)
    return nothing
end

const PARSED = argument_handling(ARGS)   # parse and validate up front (identical ARGS on every rank)
include(joinpath(pwd(), PARSED["material_file"]))         # bring dispersion/velocity/primitives/generators into scope
main(PARSED)
