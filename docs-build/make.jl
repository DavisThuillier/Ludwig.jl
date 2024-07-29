using Documenter
using Ludwig

makedocs(
    sitename="Ludwig Documentation",
    build = "../docs/",
    pages = Any[
        "Home" => "index.md",
        "Mesh" => "mesh.md"
    ]
)
