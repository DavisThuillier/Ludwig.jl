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

        # Enforce particle conservation
        for i in 1:size(L)[1]
            L[i,i] -= sum(L[i, :])
        end

        return L, k, v, E, dV, corners, corner_ids
    end
end

function assemble_collision_operator(n_ε::Int, n_θ::Int, Uee::Real, Vimp::Real, T::Real; kwargs...)
    eefile = joinpath(kwargs[:data_dir], kwargs[:material]*"_$(Float64(T))_$(n_ε)x$(n_θ).h5")
        
    L, k, v, E, dV, _, _= load(eefile)
    L *= 0.5 * Uee^2

    if Vimp != 0.0 && kwargs[:include_impurity]

        impfile = joinpath(kwargs[:data_dir], kwargs[:material]*"_"*kwargs[:imp_stem]*"imp_$(T)_$(n_ε)x$(n_θ).h5")

        Limp, _, _, _, _, _, _ = load(impfile)
        L += Limp * Vimp^2
    end

    return L, k, v, E, dV
end

function get_property(prop::String, data_dir, material::String, T::Real, n_ε::Int, n_θ::Int, Uee::Real, Vimp::Real; kwargs...)
    s = Meta.parse("get_"*prop)
    if isdefined(Main, s)
        fn = getfield(Main, s) 
        
        L, k, v, E, dV = assemble_collision_operator(n_ε, n_θ, Uee, Vimp, T; material = material, data_dir = data_dir, kwargs)
        
        return fn(L, k, v, E, dV, kb * T; kwargs)

    else
        error("Function $(s)() not defined in Main scope.")
    end
end

function write_property_to_file(prop, material, data_dir, n_ε, n_θ, Uee, Vimp, temps; include_impurity=true, addendum="", imp_stem = "")
    if addendum == ""
        file = joinpath(data_dir, prop*"_$(n_ε)x$(n_θ).dat")
    else
        file = joinpath(data_dir, prop*"_$(addendum)_$(n_ε)x$(n_θ).dat")
    end

    s = Meta.parse("get_"*prop)
    if isdefined(Main, s)
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
                q = get_property(prop, data_dir, material, T, n_ε, n_θ, Uee, Vimp, include_impurity = include_impurity, imp_stem = imp_stem)

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