module AmpProfiling

    import Flux
    import Flux.Tracker
    
    function create_model(window_size)
        linear_model = Flux.Dense(window_size, 1)
    
        non_linear_model = Flux.Chain(
            Flux.Dense(window_size, trunc(Int, window_size / 2), Flux.σ),
            Flux.Dense(trunc(Int, window_size/2), trunc(Int, window_size / 4), Flux.σ),
            Flux.Dense(trunc(Int, window_size/4), 1))
    
        #non_linear_model = Flux.Dense(trunc(Int, window_size), 1, Flux.relu)
        function model(xs)
            return linear_model(xs) .+ non_linear_model(xs)
        end
    
        return non_linear_model
    end
    
    import Permutations

    function createInputOutputSample(input, output, index, window_size)
        x = input[index:(index+window_size-1)]
        y = output[index+window_size-1]
        return x, y
    end

    function createInputOutputSamples(input, output, window_size)
        N = min(length(input), length(output)) - window_size
        p = Permutations.array(Permutations.RandomPermutation(N))

        xs = Array{Float64}(window_size,N)
        ys = Array{Float64}(1,N)

        for index in 1:N
            x,y = createInputOutputSample(input, output, index, window_size)
            xs[:,p[index]] = x
            ys[:,p[index]] = y
        end

        return xs, ys
    end

    function createInputOutputBatches(input, output, batch_size)
        N = size(input,2)

        N_batches = trunc(Int, floor(N / batch_size))

        dataset = Array{Any}(N_batches, 1)

        for index = 0:(N_batches - 1)
            dataset[index+1] = (input[:, (1 + index * batch_size):((index + 1) * batch_size)], output[:, (1 + index * batch_size):((index + 1) * batch_size)])
        end

        return dataset
    end

    function train_model(model, window_size, input, output, batch_size, number_of_epochs)
        #window_size = size(model.linear_model.W, 2)
        #window_size = size(model.W[1], 2)

        println("Creating input/output samples...")
        xs, ys = createInputOutputSamples(input, output, window_size)
    
        loss(a, b) = Flux.mse(model(a), b)
    
        N = size(xs, 2)
        println("# of input/output samples: $(N)")
    
        #opt = Flux.SGD(Flux.params(model), 0.1)
        #opt = Flux.ADAM([Flux.params(model.linear_model) ; Flux.params(model.non_linear_model)])
        opt = Flux.ADAM(Flux.params(model))
   
        evalcb() = @show(loss(xs,ys))
        
        println("Assembling batches...")
        dataset = createInputOutputBatches(xs, ys, batch_size)

        for epoch in 1:number_of_epochs
            println("Training epoch $(epoch)...")
            Flux.train!(loss, dataset, opt)
            #Flux.train!(loss, dataset, opt, cb = Flux.throttle(evalcb, 15))
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
