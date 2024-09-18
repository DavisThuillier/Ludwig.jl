module LudwigIO

export load, get_property, write_property_to_file, read_property_from_file

using HDF5
using Ludwig
using LinearAlgebra
import StaticArrays: SVector

function symmetrize(L, dV, E, T)
    fd = f0.(E, T) # Fermi dirac on grid points
    D = diagm(dV .* fd .* (1 .- fd))

    return 0.5 * (L + inv(D) * L' * D)
end

function load(file; symmetrized = false)
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

        # Enforce particle conservation
        for i in 1:size(L)[1]
            L[i,i] -= sum(L[i, :])
        end

        return L, k, v, E, dV, corners, corner_ids
    end
end

function get_property(prop::String, data_dir, material::String, T::Real, n_ε::Int, n_θ::Int, Uee::Real, Vimp::Real; include_impurity::Bool = true)

    s = Meta.parse("get_"*prop)
    if isdefined(Main, s)
        fn = getfield(Main, s) 

        eefile = joinpath(data_dir, material*"_$(Float64(T))_$(n_ε)x$(n_θ).h5")
        
        L, k, v, E, dV, _, _= load(eefile, symmetrized = true)
        L *= 0.5 * Uee^2

        if Vimp != 0.0 && include_impurity
            impfile = joinpath(data_dir, material*"_imp_$(T)_$(n_ε)x$(n_θ).h5")
            Limp, _, _, _, _, _, _ = load(impfile)
            L += Limp * Vimp^2
        end

        return fn(L, k, v, E, dV, kb * T)
    else
        error("Function $(s)() not defined in Main scope.")
    end
end

function write_property_to_file(prop, material, data_dir, n_ε, n_θ, Uee, Vimp, temps; include_impurity=true, addendum="")
    if addendum == ""
        file = joinpath(data_dir, prop*"_$(n_ε)x$(n_θ).dat")
    else
        file = joinpath(data_dir, prop*"_$(addendum)_$(n_ε)x$(n_θ).dat")
    end

    s = Meta.parse("get_"*prop)
    if isdefined(Main, s)
        fn = getfield(Main, s) 

        open(file, "w") do f
            println(f, "---")
            println(f, "quantity: $(prop)")
            println(f, "n_ε: $(n_ε)")
            println(f, "n_θ: $(n_θ)")
            println(f, "Uee: $(Uee)")
            println(f, "Vimp: $(Vimp)")
            println(f, "---")
            println(f, "# T, $(prop)")
            for T in temps
                q = get_property(prop, data_dir, material, T, n_ε, n_θ, Uee, Vimp, include_impurity = include_impurity)
                println("T = $(T), $(prop) = $(q)")
                println(f, "$(T),$(q)")
            end
        end 
    else
        error("Function $(s)() not defined in Main scope.")
    end
end

function read_property_from_file(file)
    yaml = Dict{String, Any}()
    t = Float64[]
    q = Float64[]

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
                    T, val = map(x -> parse(Float64, x), split(line, ','))
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