using CairoMakie, LaTeXStrings
using LinearAlgebra
using GLM, StatsBase, DataFrames

f1 = @formula(q ~ 1 + x + y)
model1(x, y, p) = p[1] .+ p[2] * x .+ p[3] * y 

f2 = @formula(q ~ 1 + x + y + x^2 + x*y + y^2)
model2(x, y, p) = p[1] .+ p[2] .* x .+ p[3] .* y .+ p[4] .* x.^2 .+ p[5] .* y.^2 .+ p[6] .* x .* y

f3 = @formula(q ~ 1 + x + y + x^2 + x*y + y^2 + x^3 + x^2*y + x*y^2 + y^3)
model3(x, y, p) = p[1] .+ p[2] .* x .+ p[3] .* y .+ p[4] .* x.^2 .+ p[5] .* y.^2 .+ p[6] .* x.^3 .+ p[7] .* y .^3 .+ p[8].* x .* y.+ p[9] .* x .^2 .* y .+ p[10] .* x .* y .^2

f4 = @formula(q ~ 1 + x + y + x^2 + x*y + y^2 + x^3 + x^2*y + x*y^2 + y^3 + x^4 + x^3*y + x^2 * y^2 + x * y^3 + y^4)
model4(x, y, p) = p[1] .+ p[2] .* x .+ p[3] .* y .+ p[4] .* x.^2 .+ p[5] .* y.^2 .+ p[6] .* x.^3 .+ p[7] .* y .^3 .+ p[8] .* x.^4 .+ p[9] .* y^4 .+ p[10].* x .* y.+ p[11] .* x .^2 .* y .+ p[12] .* x .* y .^2 .+ p[13] .* x^3 .* y .+ p[14] .* x^2 .* y .^2 .+ p[15] .* x .* y.^3

formulae = [f1, f2, f3, f4]
models   = [model1, model2, model3, model4]

function fit(data, q_label, q_unit, order = 1, N = 100)
    fit = lm(formulae[order], data)
    @show fit

    f = Figure(fontsize = 16, size = (700, 400))
    ax = Axis3(f[1,1], 
        azimuth = 1.8pi,
        elevation = 0.05pi,
        xlabel = L"n_\varepsilon^{-1}",
        ylabel = L"n_\theta^{-1}",
        zlabel = latexstring("$q_label")
    ) 
    # xlims!(ax, 0, maximum(x))
    # ylims!(ax, 0, maximum(y))


    x_domain = 1.5 * LinRange(0, maximum(data.x), N)
    y_domain = 1.5 * LinRange(0, maximum(data.y), N)
    fit_z = Matrix{Float64}(undef, N, N)
    color = Matrix{Tuple{Symbol, Float64}}(undef, N, N)
    fill!(color, (:gray, 0.2))
    for (i, x) in enumerate(x_domain)
        for (j, y) in enumerate(y_domain)
            fit_z[i,j] = models[order](x, y, coef(fit))
        end
    end

    q = 0
    for i in eachindex(data.q)
        if data.x[i] == 1/12 && data.y[i] == 1/38
            q = data.q[i]
        end
    end 
    @show q

    scatter!(ax, data.x, data.y, data.q, color = :black)
    scatter!(ax, 1/12, 1/38, q, color=:red)
    surface!(ax, x_domain, y_domain, fit_z, color = color)

    scatter!(ax, 0.0, 0.0, coef(fit)[1])
    text!(ax, 0.05 * maximum(data.x), 0.02 * maximum(data.y), coef(fit)[1]; text = latexstring("\$ $(q_label) = $(round(coef(fit)[1], digits = 4)) \\,\\mathrm{$(q_unit)} \$"))

    display(f)
end

function main(file)
    
    if isfile(file)
        n_ε = Int[]
        n_θ = Int[]
        q   = Float64[]

        q_label = "q"
        q_unit  = ""

        open(file, "r") do f
            for line in readlines(f)
                if startswith(line, '#')
                    continue
                elseif startswith(line, "quantity")
                    _, q_label = split(line, ':')
                elseif startswith(line, "unit")
                    _, q_unit = split(line, ':')
                else
                    x, y, z = split(line, ',')
                    push!(n_ε, parse(Int, x))
                    push!(n_θ, parse(Int, y))
                    push!(q  , parse(Float64, z))
                end
            end
        end

        data = DataFrame(x = 1 ./ n_ε, y = 1 ./ n_θ, q = q)

        indices = findall(x -> x == 10, n_ε)

        ζ = 0.7126
        f = Figure()
        ax = Axis(f[1,1])
        xlims!(ax, 0, 48)
        scatter!(ax, n_θ[indices], log.(ζ .- q[indices]))
        # scatter!(ax, n_θ[indices], q[indices] .- ζ)
        # display(f)

        # return nothing

        fit(data, q_label, q_unit, 3)
    else
        println("Please enter a valid data file.")
    end
end

data_dir = joinpath(@__DIR__, "..", "data", "Sr2RuO4", "model_2")
files = [x*"_convergence.dat" for x in 
    [   "ρ",
        #"η",
        #"λ_dxz-dyz",
        #"λ_ε"
    ]
         
]


for file in files 
    main(joinpath(data_dir, file))
end