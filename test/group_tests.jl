using Ludwig.Groups

@testset "Permutation Group" begin
    g = PermutationGroupElement([5, 4, 3, 2, 1])
    h = PermutationGroupElement([2, 4, 1, 3, 5])
    gh = PermutationGroupElement([4, 2, 5, 3, 1])

    @test g * h == gh
    @test is_identity(g * inverse(g)) == true
    @test is_identity(h * inverse(h)) == true
    @test is_identity(gh * inverse(gh)) == true

    for n in [1, 2, 3, 4, 6]
        Cₙ = Groups.get_cyclic_group(n)
        @test Groups.order(Cₙ) == n
        # Groups.get_table(Cₙ) |> display
    end

    for n in [3, 4, 6]
        Dₙ = Groups.get_dihedral_group(n)
        @test Groups.order(Dₙ) == 2n
        # Groups.get_table(Dₙ) |> display
    end

    for n in 1:6
        Sₙ = Groups.get_symmetric_group(n)
        @test Group.order(Sₙ) == factorial(n)
    end

end