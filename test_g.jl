include("AmpProfiling.jl")

import WAV

#function test()
    training_input = WAV.wavread("guitar_clean.wav")[1]
    println("# of training input samples: $(length(training_input))")
    
    training_output = WAV.wavread("guitar_processed.wav")[1]
    println("# of training output samples: $(length(training_output))")
    
    test_input = WAV.wavread("guitar_short.wav")[1][:,1]
    println("# of test samples: $(length(test_input))")

    
    nlms = 64
    lms = 1024

    unrolled_training_input = AmpProfiling.unroll_time(AmpProfiling.training_input(), nlms)
    data = zip(unrolled_training_input, training_output)
    
    batchsize = 500

    unroll_batchsize = nlms + lms + batchsize - 1

    N = min(length(training_input), length(training_output)) - (batchsize + nlms + lms -1)
    
    h = AmpProfiling.G(nlms, lms)

    loss(x, y) = Flux.mse(h(x), y)
    #opt = Flux.SGD(AmpProfiling.params(h), decay = 0.01)
    opt = Flux.ADAM(AmpProfiling.params(h))
    
    epochs = 1000
    
    for index in 1:epochs
        Flux.train!(loss, [data], opt)
        println(loss(unrolled[1][1], unrolled[1][2]))
    end
    output = zeros(length(test_input)); for n in 1:length(test_input); output[n] = Flux.Tracker.value(h(AmpProfiling.unroll_single_input(test_input, h, n)))[1]; end

#    println("Applying model to test data...")
#    test_xs = AmpProfiling.createWindowedSamples(test_input, window_size)
#    test_output = AmpProfiling.applyModel(tm, test_xs)
#    
#    println("Writing output file...")
#    WAV.wavwrite(test_output, "tm_guitar.wav", Fs=48000)
#    
#    return m
#end
