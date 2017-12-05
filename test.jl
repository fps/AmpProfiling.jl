include("AmpProfiling.jl")
import WAV

training_input = WAV.wavread("guitar_short.wav")[1][:,1]
println("# of training input samples: $(length(training_input))")

training_output = WAV.wavread("guitar_processed.wav")[1][:,1]
println("# of training output samples: $(length(training_output))")

test_input = WAV.wavread("guitar_short.wav")[1][:,1]
println("# of test samples: $(length(test_input))")

window_size = 16

m = AmpProfiling.create_model(window_size)

batch_size = 32
number_of_epochs = 150

println("Creating input/output samples...")
xs, ys = AmpProfiling.createInputOutputSamplesEWMA(training_input, training_output, window_size)

println("Permutations...")
xs, ys = AmpProfiling.randomPermuteInputOutputSamples(xs, ys)

N = size(xs, 2)
println("# of input/output samples: $(N)")

println("Assembling batches...")
dataset = AmpProfiling.createInputOutputBatches(xs, ys, batch_size)

tm = AmpProfiling.train_model(m, window_size, dataset, number_of_epochs)

println("Applying model to test data...")
test_output = AmpProfiling.applyModel(tm, test_input, window_size)

println("Writing output file...")
WAV.wavwrite(test_output, "tm_guitar.wav", Fs=48000)
