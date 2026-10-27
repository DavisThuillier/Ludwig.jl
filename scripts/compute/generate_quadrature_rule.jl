using Pkg
Pkg.activate(@__DIR__)

using ArgParse
using DelimitedFiles
using Ludwig

function argument_handling(ARGS)
    settings = ArgParseSettings(description = "Build a Brillouin-zone quadrature rule and write it to a text file.")

    @add_arg_table! settings begin
        "--T"
            help = "Temperature in Kelvin (converted to energy units via kb)"
            arg_type = Float64
            required = true
        "--width"
            help = "Width of the symmetric sampling window in units of temperature"
            arg_type = Float64
            default  = 12.0
        "--Ne"
            help = "Number of points sampled in the energy direction"
            arg_type = Int
            default = 24
        "--Ns"
            help = "Number of nodes per isoenergy surface"
            arg_type = Int
            default = 10
        "--initial_depth"
            help = "Coarse seed depth: marching seed grid has 2^initial_depth + 1 points per axis"
            arg_type = Int
            default = 7
        "--refinement_depth"
            help = "Total adaptive-subdivision depth; finest cell spans extent / 2^refinement_depth"
            arg_type = Int
            default = 24
        "--sampling_depth"
            help = "Refinement depth of the fixed coarse sampling mesh (3D only); controls the harmonic-eigensolve/candidate-node resolution independent of --refinement_depth"
            arg_type = Int
            default = 8
        "--outdir", "-o"
            help = "Output quadrature-rule directory"
            arg_type = String
            default = ""
        "--force", "-f"
            help = "Regenerate even if the output file already holds a quadrature rule"
            action = :store_true
        "material_file"
            help = "File defining dispersion, velocity, point-group generators, and lattice vectors"
            arg_type = String
            required = true
    end

    return parse_args(ARGS, settings)
end

"""
    write_quadrature_rule(path, nodes, weights, N_ibz, parsed)

Write the quadrature `nodes`/`weights` to `path` as a text file: a commented header recording the
build arguments (and `N_ibz`), a final commented row labelling the columns, then one `k…, w` row per
node.
"""
function write_quadrature_rule(path, nodes, weights, N_ibz, N, parsed)
    D = length(first(nodes))
    coordinate_labels = ("kx", "ky", "kz")[1:D]

    coordinates = Matrix{Float64}(undef, length(nodes), D)
    for i ∈ eachindex(nodes), d ∈ 1:D
        coordinates[i, d] = nodes[i][d]
    end

    open(path, "w") do io
        println(io, "# Quadrature rule")
        println(io, "# material_file       = ", parsed["material_file"])
        println(io, "# T [K]               = ", parsed["T"])
        println(io, "# width               = ", parsed["width"])
        println(io, "# energy_window [eV]  = ", parsed["width"] * kb * parsed["T"])
        println(io, "# Ne                  = ", parsed["Ne"])
        println(io, "# Ns                  = ", parsed["Ns"])
        println(io, "# initial_depth       = ", parsed["initial_depth"])
        println(io, "# refinement_depth    = ", parsed["refinement_depth"])
        println(io, "# sampling_depth      = ", parsed["sampling_depth"])
        println(io, "# N_ibz               = ", N_ibz)
        println(io, "# N                   = ", N)
        println(io, "# ", join((coordinate_labels..., "w"), ", "))
        writedlm(io, [coordinates weights], ',')   
    end

    return nothing
end

"""
    has_quadrature_data(path)

Returns `true` when `path` exists and holds at least one non-comment data row, distinguishing a
completed quadrature rule from an absent, empty, or header-only file left by an interrupted run.
"""
function has_quadrature_data(path)
    isfile(path) || return false
    open(path) do io
        for line ∈ eachline(io)
            stripped = strip(line)
            (isempty(stripped) || startswith(stripped, '#')) && continue
            return true
        end
        return false
    end
end

"""
    assert_writable(path)

Errors unless `path` can be opened for writing. Creates the file, or truncates an existing one.
"""
function assert_writable(path)
    try
        open(path, "w") do _
        end
    catch err
        error("output file $(path) is not writable: $(err)")
    end
    return nothing
end

function main(parsed)
    @info "starting generate_quadrature_rule"; flush(stderr)

    energy_window = parsed["width"] * kb * parsed["T"]   # T in Kelvin → energy units (eV)
    N_s           = parsed["Ns"]
    N_ε           = parsed["Ne"]
    initial_depth    = parsed["initial_depth"]
    refinement_depth = parsed["refinement_depth"]
    sampling_depth   = parsed["sampling_depth"]

    outdir = parsed["outdir"]
    base = "$(MATERIAL_NAME)_$(N_ε)x$(N_s)"
    outfile = joinpath(outdir, base*".quad")

    if !parsed["force"] && has_quadrature_data(outfile)
        @info "quadrature rule already present at $(outfile); skipping (pass --force to regenerate)"
        println(outfile)
        return nothing
    end
    assert_writable(outfile)   # fail fast before the long calculation

    point_group = PointGroup(generators, primitives)
    crystal     = Crystal(primitives, point_group)

    nodes, weights, residuals = Ludwig.bz_quadrature_rule(dispersion, velocity, crystal, energy_window, N_s, N_ε; initial_depth, refinement_depth, sampling_depth, progress = true)
    @show maximum(residuals; init = 0.0)

    N = length(nodes)
    N_ibz = length(nodes) ÷ order(point_group)
        

    write_quadrature_rule(outfile, nodes, weights, N_ibz, N, parsed)
    @info "wrote $(length(nodes)) nodes (N_ibz = $N_ibz) to $(outfile)"
    println(outfile)

    return nothing
end

parsed = argument_handling(ARGS)
include(joinpath(pwd(), parsed["material_file"]))   # defines dispersion/velocity/primitives/generators
main(parsed)
