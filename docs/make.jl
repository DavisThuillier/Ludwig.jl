using Documenter
using Ludwig, Ludwig.FSMesh

makedocs(
    sitename="Ludwig.jl",
    pages = Any[
        "Home" => "index.md",
        "Installation" => "install.md",
        "Mesh" => ["Overview" => "mesh/mesh.md",
        "Marching Squares" => "mesh/marching_squares.md",
        ],
        "Collision Operator" => "collision_operator.md",
        # "Example" => "example.md"
    ],
    format = Documenter.HTML(; mathengine=
        Documenter.KaTeX(
            Dict(:delimiters => [
                Dict(:left => raw"$",   :right => raw"$",   display => false),
                Dict(:left => raw"$$",  :right => raw"$$",  display => true),
                Dict(:left => raw"\[",  :right => raw"\]",  display => true),
                ],
                :macros => Dict(),
            )
        )
    )
)

deploydocs(
    repo = "github.com/DavisThuillier/Ludwig.jl.git",
    push_preview = false
)
