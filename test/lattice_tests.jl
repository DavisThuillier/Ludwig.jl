using Ludwig.Lattices

@testset "Lattice Constructors" begin
    a1 = [1.0, 1.0]
    a2 = [1.0, -2.0]
    A  = [1.0 1.0; 1.0 -2.0]

    @test Lattice(a1, a2) == Lattice(A)
end

@testset "Lattice Types" begin
    obl_lat = Lattice([1.0 0.0; 0.2 -0.5])
    @show Lattices.get_bz(obl_lat)
    @test lattice_type(obl_lat) == "Oblique"

    sqr_lat = Lattice([1.0 0.0; 0.0 1.0])
    @show Lattices.get_bz(sqr_lat)
    @test lattice_type(sqr_lat) == "Square"

    rec_lat = Lattice([1.0 0.0; 0.0 2.0])
    @show Lattices.get_bz(rec_lat)
    @test lattice_type(rec_lat) == "Rectangular"

    hex_lat = Lattice([1.0 -0.5; 0.0 sqrt(3)/2.0])
    @show Lattices.get_bz(hex_lat)
    @test lattice_type(hex_lat) == "Hexagonal"

end