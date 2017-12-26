include("AmpProfiling.jl")

import WAV

#function test()
    input_range = 1:2500
    training_input = WAV.wavread("guitar_short.wav")[1][input_range,1]
    println("# of training input samples: $(length(training_input))")
    
    training_output = WAV.wavread("guitar_processed.wav")[1][input_range,1]
    println("# of training output samples: $(length(training_output))")
    
    test_input = WAV.wavread("guitar_short.wav")[1][:,1]
    println("# of test samples: $(length(test_input))")
    
    nlms = 32
    lms = 512
    
    h = AmpProfiling.H(nlms, lms)

    unrolled = AmpProfiling.unroll(training_input, h, training_output)
     
    loss(x, y) = Flux.mse(h(x), y)
    #opt = Flux.SGD(AmpProfiling.params(h), decay = 0.01)
    opt = Flux.ADAM(AmpProfiling.params(h))
    
    for index in 1:500
        p = Permutations.array(Permutations.RandomPermutation(length(unrolled)))
        Flux.train!(loss, unrolled[p], opt)
        println(loss(unrolled[100][1], unrolled[100][2]))
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
