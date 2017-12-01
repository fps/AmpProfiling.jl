import Flux
import Flux.Tracker

function create_model(window_size)
    linear_model = Flux.Dense(window_size, 1)

    non_linear_model = Flux.Chain(
        Flux.Dense(window_size, trunc(Int, window_size / 2), Flux.relu),
        Flux.Dense(trunc(Int, window_size/2), trunc(Int, window_size / 4), Flux.relu),
        Flux.Dense(trunc(Int, window_size/4), trunc(Int, window_size / 8), Flux.relu),
        Flux.Dense(trunc(Int, window_size/8), 1, Flux.relu))

    struct model_model
        linear_model
        non_linear_model
    end

    

    function model(xs) 
        return linear_model(xs) .+ non_linear_model(xs)
    end

    return model
end
