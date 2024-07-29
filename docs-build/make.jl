using Documenter
using Ludwig

makedocs(
    sitename="Ludwig",
    build = "../docs/",
    pages = Any[
        "Home" => "index.md",
        "Installation" => "install.md",
        "Mesh" => ["Marching Squares" => "mesh/marching_squares.md",
        "Single Band Mesh" => "mesh/mesh_sb.md",
        "Multiband Meshes"=> "mesh/mesh_mb.md",
        ]
    ]
)