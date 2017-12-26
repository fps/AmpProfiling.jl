include("AmpProfiling.jl")

import WAV

#function test()
    training_input = WAV.wavread("guitar_clean.wav")[1]
    println("# of training input samples: $(length(training_input))")
    
    training_output = WAV.wavread("guitar_processed.wav")[1]
    println("# of training output samples: $(length(training_output))")
    
    test_input = WAV.wavread("guitar_short.wav")[1][:,1]
    println("# of test samples: $(length(test_input))")
    
    nlms = 32
    lms = 512
    
    batchsize = 1500

    N = min(length(training_input), length(training_output)) - (batchsize + nlms + lms -1)
    
    h = AmpProfiling.H(nlms, lms)

    loss(x, y) = Flux.mse(h(x), y)
    #opt = Flux.SGD(AmpProfiling.params(h), decay = 0.01)
    opt = Flux.ADAM(AmpProfiling.params(h))
    
    epochs = 100
    
    for index in 1:epochs
        sindex = randperm(N)[1]
        eindex = sindex + batchsize
        unrolled = AmpProfiling.unroll(training_input[sindex:eindex], h, training_output[sindex:eindex])
     
        p = Permutations.array(Permutations.RandomPermutation(length(unrolled)))
        Flux.train!(loss, unrolled[p], opt)
        println(loss(unrolled[1][1], unrolled[1][2]))
    end
#    println("Applying model to test data...")
#    test_xs = AmpProfiling.createWindowedSamples(test_input, window_size)
#    test_output = AmpProfiling.applyModel(tm, test_xs)
#    
#    println("Writing output file...")
#    WAV.wavwrite(test_output, "tm_guitar.wav", Fs=48000)
#    
#    return m
#end
