using Ludwig.Lattices
using Ludwig.MarchingSquares

@testset "Lattice Constructors" begin
    a1 = [1.0, 1.0]
    a2 = [1.0, -2.0]
    A  = [1.0 1.0; 1.0 -2.0]

    @test Lattice(a1, a2) == Lattice(A)
end

@testset "Lattice Types" begin
    obl_lat = Lattice([1.0 0.0; 0.2 -0.5])
    @test lattice_type(obl_lat) == "Oblique"

    sqr_lat = Lattice([1.0 0.0; 0.0 1.0])
    @test lattice_type(sqr_lat) == "Square"

    rec_lat = Lattice([1.0 0.0; 0.0 2.0])
    @test lattice_type(rec_lat) == "Rectangular"

    hex_lat = Lattice([1.0 -0.5; 0.0 sqrt(3)/2.0])
    @test lattice_type(hex_lat) == "Hexagonal"

end

@testset "Brillouin Zone Mapping" begin
    hex_lat = Lattice([1.0 -0.5; 0.0 sqrt(3)/2.0])
    bz = Lattices.get_bz(hex_lat)
    rlv = Lattices.reciprocal_lattice_vectors(hex_lat)
    x_range, y_range = MarchingSquares.get_bounding_box(bz)

    x_range = 2 .* x_range
    y_range = 2 .* y_range

    for i in 1:10000
        kx = x_range[1] + (x_range[2] - x_range[1]) * Random.rand(Float64)
        ky = y_range[1] + (y_range[2] - y_range[1]) * Random.rand(Float64)
        k = Lattices.map_to_bz([kx, ky], bz, rlv)
        @test Lattices.in_polygon(k, bz)
    end
end
