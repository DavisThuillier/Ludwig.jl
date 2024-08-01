function multiband_weight(p, F, N)
    k = LinRange(-0.5, 0.5, N)
    n_bands = length(p.w)

    weights = Array{Float64}(undef, N, N, n_bands)
    for i in 1:N, j in 1:N
        for μ in 1:n_bands
            weights[i,j, μ] = F(p, [k[i], k[j]], μ)
        end 
    end

    return weights
end