include("AmpProfiling.jl")
import WAV
import Plots

# Plots.plotly()

sampling_rate = 48000

training_input = sin.((2.0 * pi * 100) .* (0:(1.0/sampling_rate):1.0))
println(length(training_input))

training_output = 0.4 .* training_input
println(length(training_output))

test_input = training_input[1:trunc(Int, floor(sampling_rate/2.0))]
println(length(test_input))

window_size = 32

m = AmpProfiling.create_model(window_size)

batch_size = 16
number_of_batches = 5000

tm = AmpProfiling.train_model(m, window_size, training_input, training_output, batch_size, number_of_batches)

moutput = (Float64)[]
for index in 1:(length(test_input) - window_size)
    append!(moutput, Flux.Tracker.value(tm(test_input[index:(index + (window_size - 1))])))
end

# Plots.plot(moutput)

WAV.wavwrite(moutput, "tm_guitar.wav", Fs=48000)
