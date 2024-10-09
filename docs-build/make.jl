using Documenter
using Ludwig

# Types from other packages needed in docs
import Interpolations: ScaledInterpolation 

makedocs(
    sitename="Ludwig.jl",
    build = "../docs/",
    pages = Any[
        "Home" => "index.md",
        "Installation" => "install.md",
        "Mesh" => ["Overview" => "mesh/mesh.md",
        "Marching Squares" => "mesh/marching_squares.md",
        "Single Band Mesh" => "mesh/mesh_sb.md",
        "Multiband Meshes"=> "mesh/mesh_mb.md",
        ],
        "Collision Operator" => "collision_operator.md",
        "Example" => "example.md"
    ]
)