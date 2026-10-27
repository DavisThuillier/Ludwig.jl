###
### Ludwig test suite
###
#
# `Pkg.test()` runs every domain file listed below. To run a single domain, either invoke
# its file directly
#
#     julia --project=. test/mesh_test.jl
#
# or pass a filter, matched against the file names:
#
#     julia --project=. test/runtests.jl mesh
#     Pkg.test("Ludwig"; test_args = ["mesh"])
#
# Each domain file carries its own `using` header and includes the fixtures it needs, so it
# runs standalone. A full-suite run shares one `Main`, so an import a domain file forgot to
# declare stays hidden until that file is run on its own.

using Test

# Domain files, in the dependency order of the sources they cover.
const TEST_FILES = [
    "crystal_symmetry_test.jl",
    "representations_test.jl",
    "polytope_test.jl",
    "mesh_test.jl",
    "marching_simplices_test.jl",
    "sampling_mesh_test.jl",
    "quadrature_test.jl",
    "collision_operator_test.jl",
    "processing_test.jl",
    "transport_test.jl",
]

selected = isempty(ARGS) ? TEST_FILES :
           filter(file -> any(pattern -> occursin(pattern, file), ARGS), TEST_FILES)
isempty(selected) &&
    error("no test file matches $(ARGS); available: $(join(TEST_FILES, ", "))")

@testset verbose = true "Ludwig" begin
    for file in selected
        @testset "$file" begin
            include(file)
        end
    end
end
