include("AmpProfiling.jl")
import WAV

training_input = WAV.wavread("guitar_clean.wav")[1][:,1]
println(length(training_input))

training_output = WAV.wavread("guitar_processed.wav")[1][:,1]
println(length(training_output))

test_input = WAV.wavread("guitar_short.wav")[1][:,1]
println(length(test_input))

window_size = 32

m = AmpProfiling.create_model(window_size)

batch_size = 32
number_of_batches = 1000

tm = AmpProfiling.train_model(m, window_size, training_input, training_output, batch_size, number_of_batches)

moutput = (Float64)[]
for index in 1:(size(test_input, 1) - window_size)
    append!(moutput, Flux.Tracker.value(tm(test_input[index:(index + (window_size - 1))])))
end

WAV.wavwrite(moutput, "tm_guitar.wav", Fs=48000)
