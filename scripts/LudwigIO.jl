module LudwigIO

export load, get_property, write_property_to_file, read_property_from_file

using HDF5
using Ludwig
using LinearAlgebra
import StaticArrays: SVector

function load(file; symmetrized = true)
    h5open(file, "r") do fid
        g = fid["data"]

        L = 2pi * read(g, "L") 
        k = reinterpret(SVector{2,Float64}, vec(transpose(read(g, "momenta"))))
        v = reinterpret(SVector{2,Float64}, vec(transpose(read(g, "velocities"))))
        E = read(g, "energies")
        dV = read(g, "dVs")
        corners = read(g, "corners")
        corner_ids = read(g, "corner_ids")
        T = read(g, "T")

        symmetrized && (L = symmetrize(L, dV, E, kb * T))

        return L, k, v, E, dV, corners, corner_ids
    end
end

function assemble_collision_operator(n_ε::Int, n_θ::Int, Uee::Real, Vimp::Real, T::Real; symmetrized = true, kwargs...)
    eefile = kwargs[:addendum] == "" ? kwargs[:material] : kwargs[:material]*"_"*kwargs[:addendum]
    eefile = joinpath(kwargs[:data_dir], eefile*"_$(Float64(T))_$(n_ε)x$(n_θ).h5")
        
    L, k, v, E, dV, _, _= load(eefile; symmetrized = symmetrized)
    L *= 0.5 * Uee^2

    if Vimp != 0.0 && kwargs[:include_impurity]

        impfile = joinpath(kwargs[:data_dir], kwargs[:material]*"_"*kwargs[:imp_stem]*"imp_$(T)_$(n_ε)x$(n_θ).h5")

        Limp, _, _, _, _, _, _ = load(impfile, symmetrized = symmetrized)
        L += Limp * Vimp^2
    end

    # Enforce particle conservation
    for i in 1:size(L)[1]
        L[i,i] -= sum(L[i, :])
    end

    return L, k, v, E, dV
end

function get_property(prop::String, data_dir::String, material::String, n_ε::Int, n_θ::Int, Uee::Real, Vimp::Real; symmetrized = true, kwargs...)
    T = kwargs[:T]
    s = Meta.parse("get_"*prop)
    if isdefined(Main, s)
        fn = getfield(Main, s) 
        
        L, k, v, E, dV = assemble_collision_operator(n_ε, n_θ, Uee, Vimp, T; material = material, data_dir = data_dir, symmetrized = symmetrized, kwargs...)
        
        return fn(L, k, v, E, dV, kb * T; kwargs)

    else
        error("Function $(s)() not defined in Main scope.")
    end
end

function write_property_to_file(X::String, Y::String, X_range, material::String, data_dir::String, n_ε::Int, n_θ::Int, Uee::Float64, Vimp::Float64; addendum = "", include_impurity = true, imp_stem = "", kwargs...)
    if addendum == ""
        file = joinpath(data_dir, Y*"_"*X*"-series.dat")
    else
        file = joinpath(data_dir, Y*"_$(addendum)_"*X*"-series.dat")
    end
    
    s = Meta.parse("get_"*Y)
    if isdefined(Main, s)
        open(file, "w") do f
            println(f, "---")
            println(f, "quantity: $(Y)")
            println(f, "n_ε: $(n_ε)")
            println(f, "n_θ: $(n_θ)")
            println(f, "Uee: $(Uee)")
            println(f, "Vimp: $(Vimp)")
            if X != "T"
                T = kwargs[:T]
                println(f, "T: $(T)")
            end
            println(f, "---")
            println(f, "# $(X),$(Y)")
            for x in X_range

                if X != "T"
                    exp = Meta.parse(
                        "get_property(\"$Y\", $(repr(data_dir)),  \"$material\", $n_ε, $n_θ, $Uee, $Vimp; include_impurity = $include_impurity, $(X) = $(x), addendum = \"$addendum\", imp_stem = \"$imp_stem\", T = $T, $(kwargs...))"
                    )
                else
                    exp = Meta.parse(
                        "get_property(\"$Y\", $(repr(data_dir)),  \"$material\", $n_ε, $n_θ, $Uee, $Vimp; include_impurity = $include_impurity, $(X) = $(x), addendum = \"$addendum\", imp_stem = \"$imp_stem\", $(kwargs...))"
                    )
                end

                y = eval(exp)
            
                println("$(X) = $(x), $(Y) = $(y)")
                println(f, "$(x),$(y)")
            end
        end 
    else
        error("Function $(s)() not defined in Main scope.")
    end
end

function parity(v, n_bands)
    ℓ = length(v) ÷ n_bands

    parities = Vector{Float64}(undef, n_bands)
    for μ in 1:n_bands
        parities[μ] = single_band_parity(v[(μ-1)*ℓ + 1:μ*ℓ])
        # for i in 1:(ℓ ÷ 2)
        #     p += sign(v[ℓ * (μ - 1) + i] * v[ℓ * (μ - 1) + mod(i - 1 + ℓ÷2, ℓ) + 1])
        # end
    end

    # return p / (3*ℓ)
    return round(Int, sum(parities) / n_bands) 
end

function single_band_parity(v)
    ℓ = length(v)

    p = 0.0
    for i in 1:(ℓ ÷ 2)
        p += sign(real(v[i] * v[mod(i - 1 + ℓ÷2, ℓ) + 1]))
    end

    return 2 * p / ℓ 
end

function write_spectrum_to_file(material, n_bands, data_dir, n_ε, n_θ, Uee, Vimp, T; include_impurity=true, addendum="", imp_stem = "")
    if addendum == ""
        file = joinpath(data_dir, "spectrum_$(T).dat")
    else
        file = joinpath(data_dir, "spectrum_$(addendum)_$(T).dat")
    end

    L, k, _, E, dV = assemble_collision_operator(n_ε, n_θ, Uee, Vimp, T; material = material, data_dir = data_dir, include_impurity=include_impurity, imp_stem=imp_stem)

    fd = f0.(E, kb * T) # Fermi dirac on grid points
    W = diagm(dV .* fd .* (1 .- fd))
    D = diagm(1 ./ sqrt.(dV .* fd .* (1 .- fd)))
    Ls = 0.5 * (W * L + L' * W) 

    # Enforce particle conservation
    for i in eachindex(k)
        Ls[i,i] -= sum(Ls[:, i])
    end

    M = Hermitian(D * Ls * D)
    
    @time eigenvectors = eigvecs(M)
    @time eigenvalues = eigvals(M) / hbar

    parities = Vector{Float64}(undef, length(eigenvalues))
    for i in eachindex(parities)
        parities[i] = parity(eigenvectors[:, i], n_bands)
    end

    open(file, "w") do f
        println(f, "---")
        println(f, "quantity: spectrum (s^{-1})")
        println(f, "n_ε: $(n_ε)")
        println(f, "n_θ: $(n_θ)")
        println(f, "Uee: $(Uee)")
        println(f, "Vimp: $(Vimp)")
        println(f, "T: $(T)")
        println(f, "---")
        println(f, "# λ, Parity")
        for i in eachindex(eigenvalues)
            println(f, "$(eigenvalues[i]), $(parities[i])")
        end
    end 
end

function read_property_from_file(file)
    yaml = Dict{String, Any}()
    t = Float64[]
    q = []

    open(file, "r") do f
        inyaml = false
        for line in eachline(f)
            if !startswith(line, "---")
                if inyaml
                    key, prop = split(line, ':')
                    if tryparse(Float64, prop) !== nothing
                        num_prop = parse(Float64, prop)
                        isinteger(num_prop) && (num_prop = Int(num_prop))
                        yaml[key] = num_prop
                    else
                        yaml[key] = strip(prop)
                    end
                else 
                    startswith(line, "#") && continue
                    T, val = map(x -> parse(ComplexF64, x), split(line, ','))
                    imag(val) == 0.0 && (val = real(val))
                    push!(t, T)
                    push!(q, val)
                end
            end
            startswith(line, "---") && (inyaml = !inyaml)
        end
    end

    return t, q, yaml
end

end