using Documenter
using Ludwig, Ludwig.FSMesh

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
        "Collision Operator" => "collision_operator.md",
        "Transport Properties" => [
            "Inner Product" => "properties/inner_product.md",
            "Transport Coefficients" => "properties/transport.md",
        ],
    ],
)

deploydocs(
    repo = GitHub("DavisThuillier", "Ludwig.jl"),
    # repo = "github.com/DavisThuillier/Ludwig.jl.git",
    push_preview = false
)
