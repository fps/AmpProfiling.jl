include("AmpProfiling.jl")

import WAV

#function test()
    println("loading data...")
    training_input = WAV.wavread("guitar_clean.wav")[1]
    println("# of training input samples: $(length(training_input))")
    
    training_output = WAV.wavread("guitar_processed.wav")[1]
    println("# of training output samples: $(length(training_output))")
    
    test_input = WAV.wavread("guitar_short.wav")[1][:,1]
    println("# of test samples: $(length(test_input))")

    
    println("unrolling data...")
    nlms = 64
    lms = 1024

    N = min(length(training_input), length(training_output)) - (nlms + lms +1)

    unrolled_training_input = AmpProfiling.unroll_time(AmpProfiling.unroll_time2(AmpProfiling.training_input(), nlms), lms)

    # unrolled_training_input = AmpProfiling.unroll_time2(AmpProfiling.unroll_time(AmpProfiling.training_input(), nlms), lms)
    data = collect(zip(unrolled_training_input[1:N], training_output[(nlms+lms-1).+(1:N)]))
    
    batchsize = 500

    unroll_batchsize = nlms + lms + batchsize - 1

    
    h = AmpProfiling.G(nlms, lms)

    loss(x, y) = Flux.mse(h(x), y)
    #opt = Flux.SGD(AmpProfiling.params(h), decay = 0.01)
    opt = Flux.ADAM(AmpProfiling.params(h))
    
    epochs = 10000
    
    println("training...")
    for index in 1:epochs
        println(index)
        p = randperm(N)[1:batchsize]
        Flux.train!(loss, data[p], opt)
        println(loss(data[1][1], data[1][2]))
    end
    # output = zeros(length(test_input)); for n in 1:length(test_input); output[n] = Flux.Tracker.value(h(AmpProfiling.unroll_single_input(test_input, h, n)))[1]; end

    println("generating output...")
    output = zeros(length(test_input)); for n in 1:length(test_input); output[n] = h(unrolled_training_input[n]).data[1]; end

    println("writing output file...")
    WAV.wavwrite(output, "tm_guitar.wav", Fs=48000)

#    println("Applying model to test data...")
#    test_xs = AmpProfiling.createWindowedSamples(test_input, window_size)
#    test_output = AmpProfiling.applyModel(tm, test_xs)
#    
#    println("Writing output file...")
#    WAV.wavwrite(test_output, "tm_guitar.wav", Fs=48000)
#    
#    return m
#end
