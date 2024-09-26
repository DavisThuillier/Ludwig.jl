function Weff_squared_123(p1::Patch, p2::Patch, p3::Patch, Fpp::Function, Fpk::Function, k4, μ4)
    f13 = Fpp(p1, p3)
    f23 = Fpp(p2, p3)
    f14 = Fpk(p1, k4, μ4)
    f24 = Fpk(p2, k4, μ4)

    return abs2(f13*f24 - f14*f23) + 2 * abs2(f14*f23)

end

function Weff_squared_124(p1::Patch, p2::Patch, p4::Patch, Fpp::Function, Fpk::Function, k3, μ3)
    f14 = Fpp(p1, p4)
    f24 = Fpp(p2, p4)

    f13 = Fpk(p1, k3, μ3)
    f23 = Fpk(p2, k3, μ3)

    return abs2(f13*f24 - f14*f23) + 2 * abs2(f14*f23)
end