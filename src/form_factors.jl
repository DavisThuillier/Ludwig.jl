function multiband_weight(p, H, N)
    k = LinRange(-0.5, 0.5, N)
    n_bands = length(p.w)
    W = MMatrix{n_bands, n_bands, Float64}(undef) # Preallocation

    weights = Array{ComplexF64}(undef, N, N, n_bands)
    for i in 1:N, j in 1:N
        W .= eigvecs(H)
        for μ in 1:n_bands
            weights[i,j, μ] = dot(p.w, W[:, μ])
        end 
    end

    return weights
end