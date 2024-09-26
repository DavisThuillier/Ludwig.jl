using HDF5

function main()
    data_dir = joinpath(@__DIR__, "..", "data", "Sr2RuO4", "model_2")

    files = filter(x -> endswith(x, "h5"), readdir(data_dir))
    for f in files
        T = parse(Float64, split(f, "_")[end-1])
        h5open(joinpath(data_dir, f), "r+") do fid
            g = fid["data"]
            if !("T" ∈ keys(g))
                g["T"] = T
            end
        end
    end

end

main()