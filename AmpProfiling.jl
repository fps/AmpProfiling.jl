module AmpProfiling

    import Flux
    import Flux.Tracker
    
    struct Convolutional
        f
        N
        in
    end

    #Convolutional(f, in::Integer, out::Integer, N::Integer) =
    #    Convolutional(f, N, in, out)

    # Overload call, so the object can be used as a function
    function (m::Convolutional)(x)
        ret = Flux.Tracker.TrackedArray(zeros(m.N))
        for n = 1:m.N
            mask = zeros(m.N)
            mask[n] = 1
            in_start  = n
            in_end    = n + m.in - 1
            
            input = x[in_start:in_end]
            
            ret = ret .+ (mask .* m.f(input))
        end
        return ret
    end

    function create_non_linear_model(window_size)
        return Flux.Chain(
            Flux.Dense(window_size, trunc(Int, window_size / 2), Flux.relu),
            Flux.Dense(trunc(Int, window_size/2), trunc(Int, window_size / 4), Flux.relu),
            Flux.Dense(trunc(Int, window_size/4), 1))
    end

    function create_linear_model(window_size)
        return Flux.Dense(window_size, 1)
    end

    struct G
        W
        G(i, o) = new(Flux.param(Flux.initn(i, o)))
    end
    
    (g::G)(fs) = g.W' * fs
    
    function create_layered_model(window_size1, window_size2)
        f = create_non_linear_model(window_size1);
        g = G(window_size2, 1)
        return g
    end
    
    
    function create_unrolled_model(window_size1, window_size2)
        
    end

    """
        Creates a model that operates on an input 
        window of size `window_size`
    """
    function create_model(window_size)
        linear_model = Flux.Dense(window_size, 1)
    
    
        non_linear_model = Flux.Chain(
            Flux.Dense(window_size, trunc(Int, window_size / 2), Flux.relu),
            Flux.Dense(trunc(Int, window_size/2), trunc(Int, window_size / 4), Flux.relu),
            Flux.Dense(trunc(Int, window_size/4), 1))

        #non_linear_model = Flux.Chain(
        #    Flux.Dense(window_size, trunc(Int, window_size / 2), Flux.σ),
        #    Flux.Dense(trunc(Int, window_size/2), trunc(Int, window_size / 4), Flux.σ),
        #    Flux.Dense(trunc(Int, window_size/4), 1))
        #non_linear_model = Flux.Dense(trunc(Int, window_size), 1, Flux.relu)
        function model(xs)
            return linear_model(xs) .+ non_linear_model(xs)
        end
    
        return non_linear_model
    end
    
    import Permutations

    function createWindowedSamples(input, window_size)
        N = length(input) - window_size
        ret = Array{Float64}(window_size, N)

        for index in 1:N
            ret[:,index] = input[index:(index+window_size-1)]
        end

        return ret
    end

    function createInputOutputSample(input, output, index, window_size)
        x = input[index:(index+window_size-1)]
        y = output[index+window_size-1]
        return x, y
    end

    """
        Takes input and output sequences and expands the inputs
        so that each input vector contains all samples in the
        `window_size`
    """
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

    function createEWMASamples(input, window_size, coefficient)
        N = length(input)
        xs = Array{Float64}(window_size,N)

        coefficients = zeros(window_size,1)
        for dim in 1:window_size
            coefficients[dim, 1] = coefficient^(dim-1)
        end

        for dim in 1:window_size
            xs[dim, 1] = input[1]
        end

        for index in 2:N 
            for dim in 1:window_size
                xs[dim, index] = coefficients[dim, 1] * input[index] + (1.0 - coefficients[dim, 1]) * xs[dim, index-1]
            end
        end

        return xs
    end

    function createInputOutputSamplesEWMA(input, output, window_size, coefficient)
        xs = createEWMASamples(input, window_size, coefficient)
        ys = output'

        return xs, ys
    end

    function randomPermuteInputOutputSamples(input, output)
        xs = input
        ys = output

        N = size(xs, 2)

        p = Permutations.array(Permutations.RandomPermutation(N))

        xs = xs[:, p]
        ys = ys[:, p]

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

    function train_model(model, window_size, dataset, number_of_epochs)
        loss(a, b) = Flux.mse(model(a), b)
    
        #opt = Flux.SGD(Flux.params(model), 0.1)
        #opt = Flux.ADAM([Flux.params(model.linear_model) ; Flux.params(model.non_linear_model)])
        opt = Flux.ADAM(Flux.params(model))
   
        evalcb() = @show(loss(dataset[1][1],dataset[1][2]))
        
        for epoch in 1:number_of_epochs
            println("Training epoch $(epoch)...")
            #Flux.train!(loss, dataset, opt)
            Flux.train!(loss, dataset, opt, cb = Flux.throttle(evalcb, 1))
        end
        
        return model
    end

    function applyModel(model, input)
        return Flux.Tracker.value(model(input))'
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
    
    function createCPPCode(model, name)
        includes = """
            #pragma once
            #include <Eigen/Dense>
            #include <vector>
            """

        parameters = """
                std::vector<Eigen::MatrixXd> W;
                std::vector<Eigen::MatrixXd> b;
            """

        constructor_body_parts = Array{String}(length(model.layers), 1)

        for layer in 1:length(model.layers)
            rows = size(model.layers[layer].W, 1)
            cols = size(model.layers[layer].W, 2)

            creation = """
                        W.push_back(Eigen::MatrixXd($(rows), $(cols)));
                        b.push_back(Eigen::MatrixXd($(rows), 1));
                """

            initialization_parts_W = Array{String}(rows,cols)
            initialization_parts_b = Array{String}(rows)
            for row in 1:rows
                initialization_parts_b[row] = """
                            b[$(layer-1)]($(row-1), 0) = $(Flux.Tracker.value(model.layers[layer].b[row]));
                    """
                for col in 1:cols
                    initialization_parts_W[row, col] = """
                                W[$(layer-1)]($(row-1), $(col-1)) = $(Flux.Tracker.value(model.layers[layer].W[row, col]));
                        """
                end
            end

            constructor_body_parts[layer] = """
                        // Layer $(layer-1)
                $(creation)
                $(string(initialization_parts_W...))
                $(string(initialization_parts_b...))
                """
        end
        

        constructor = """
                $(name)()
                {
            $(string(constructor_body_parts...))
                }
            """

        struct_body = """
            $(parameters)
            $(constructor)
            """

        cppstruct = """
            struct $(name)
            {
            $(struct_body)
            };
            """

        cpp = """
            $(includes)
            $(cppstruct)
            """
        return cpp

    end
   
end

