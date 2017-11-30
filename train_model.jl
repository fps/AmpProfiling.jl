import Flux
import Flux.Tracker
import Permutations

function train_model(model, test, processed, batch_size, number_of_batches)
    window_size = size(model.linear_model.W, 2)

    loss(xs, ys) = Flux.mse(model(xs), ys)

    test_N = size(test, 1)
    proc_N = size(processed, 1)

    N = min(test_N, proc_N) - window_size
    println(N)

    #opt = Flux.SGD(Flux.params(model), 0.001)
    opt = Flux.ADAM(Flux.params(model))

    for batch in 1:number_of_batches
        p = Permutations.array(Permutations.RandomPermutation(N))[1:batch_size]
        #println(p)

        xs = Array{Float64}(window_size,0)
        ys = Array{Float64}(1,0)
        for index in 1:batch_size
            # println(p[index])
            x = test[p[index]:(p[index] + window_size - 1)]
            y = processed[p[index] + window_size - 1]
            xs = [xs x]
            ys = [ys y]
        end

        #println(model(xs))
        #println(size(xs))
        #println(size(ys))

        data = [(xs, ys)]
        Flux.train!(loss, data, opt)  
        println(loss(xs, ys))
    end

    return model
end
