@testset "Lattice Constructors" begin
    a1 = [1.0, 1.0]
    a2 = [1.0, -2.0]
    A  = [1.0 1.0; 1.0 -2.0]

    @test Lattices.primitives(Lattices.Lattice(a1, a2)) == A
end

@testset "Lattice Types" begin
    @test Lattices.lattice_type(Lattices.Lattice([1.0 0.0; 0.0 1.0])) == "Square"
end