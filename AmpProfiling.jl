module AmpProfiling

    import Flux
    import Flux.Tracker
    
    function create_model(window_size)
        linear_model = Flux.Dense(window_size, 1)
    
        non_linear_model = Flux.Chain(
            Flux.Dense(window_size, trunc(Int, window_size / 2), Flux.relu),
            Flux.Dense(trunc(Int, window_size/2), trunc(Int, window_size / 4), Flux.relu),
            Flux.Dense(trunc(Int, window_size/4), trunc(Int, window_size / 8), Flux.relu),
            Flux.Dense(trunc(Int, window_size/8), 1, Flux.relu))
    
        function model(xs)
            return linear_model(xs) .+ non_linear_model(xs)
        end
    
        return model
    end
    
    import Permutations
    
    function train_model(model, test, processed, batch_size, number_of_batches)
        window_size = size(model.linear_model.W, 2)
        #window_size = size(model.W, 2)
    
        loss(xs, ys) = Flux.mse(model(xs), ys)
    
        test_N = size(test, 1)
        proc_N = size(processed, 1)
    
        N = min(test_N, proc_N) - window_size
        println(N)
    
        #opt = Flux.SGD(Flux.params(model), 0.001)
        opt = Flux.ADAM([Flux.params(model.linear_model) ; Flux.params(model.non_linear_model)])
    
        for batch in 1:number_of_batches
            println(batch)
            p = Permutations.array(Permutations.RandomPermutation(N))[1:batch_size]
            #println(p)
    
            xs = Array{Float64}(window_size,0)
            ys = Array{Float64}(1,0)
            @timed for index in 1:batch_size
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
            @timed Flux.train!(loss, data, opt)
            println(loss(xs, ys))
        end
    
        return model
    end

    function generate_test_sound(sampling_rate, random_seed)
        # make it reproducible
        srand(random_seed)
    
        out = Array{Float64}(1)
    
        # append a single dirac pulse after 10 samples so we can check
        # with the processed signal later
        append!(out, zeros(10))
        append!(out, ones(1))
        append!(out, zeros(10))
    
        # add a pause of 0.1 secs
        pause_seconds = 0.1
        pause = zeros(1, (Int64)(pause_seconds * sampling_rate))
    
        append!(out, pause)
    
        # generate random dirac pulses to probe the dynamic range
        # of the processor
        for index in 1:1000
            append!(out, 2.0 * rand(1) - 1)
    
            pause_seconds = 0.01 * rand(1)[1]
            pause = zeros(1, trunc(Int, pause_seconds * sampling_rate))
    
            append!(out, pause)
        end
    
        # and generate random frequency blips to probe some more
        # for freq in 50.0:50.0:(sampling_rate/4.0)
        for index in 1:1000
            freq = abs(sampling_rate / 8.0) * randn(1)[1]
            #freq = (sampling_rate / 2.0) * rand(1)[1]
            #freq = 100.0
            #append!(out, sin.(0:(2.0*pi*freq/sampling_rate):(sampling_rate / freq))')
    
            gain = rand(1)[1]
            append!(out, gain .* sin.(freq * 2.0 * pi .* (0:(1.0/sampling_rate):(1.0/freq)))')
            append!(out, zeros(trunc(Int, 0.01 * sampling_rate)))
        end
    
        return reshape(out, length(out), 1)
    end
    
    
end
