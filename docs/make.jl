using Documenter
using Ludwig

makedocs(
    sitename="Ludwig.jl",
    pages = Any[
        "Home" => "index.md",
        "Installation" => "install.md",
        "Tutorials" => [
            "Mesh Generation" => "tutorials/mesh_generation.md",
            "Simple Calculation" => "tutorials/simple_calculation.md",
        ],
        "Mesh" => [
            "Overview" => "mesh/mesh.md",
            "Marching Squares" => "mesh/marching_squares.md",
        ],
        "Collision Operator" => [
            "Overview" => "collision_operator.md",
            "Pournin Volume" => "collision_operator/pournin_volume.md",
        ],
        "Transport Properties" => [
            "Overview" => "properties/transport.md",
            "Units and SI Conversion" => "properties/units.md",
        ],
        "API Reference" => "api.md",
    ],
)

deploydocs(
    repo = "github.com/DavisThuillier/Ludwig.jl.git",
    devbranch = "develop",
    devurl = "dev",
    versions = ["stable" => "v^", "dev" => "dev", "v0.3.0", "v0.2.2", "v0.2.1"],
    push_preview = false
)
