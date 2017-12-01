include("AmpProfiling.jl")
import WAV

guitar_clean = WAV.wavread("guitar_clean.wav")[1][:,1]
guitar_processed = WAV.wavread("guitar_processed.wav")[1][:,1]
guitar_short = WAV.wavread("guitar_short.wav")[1][:,1]

window_size = 32

m = AmpProfiling.create_model(window_size)

batch_size = 32
number_of_batches = 10000

tm = AmpProfiling.train_model(m, guitar_clean, guitar_processed, batch_size, number_of_batches)

moutput = (Float64)[]
for index in 1:(size(guitar_short, 1) - window_size)
    append!(moutput, Flux.Tracker.value(tm(guitar_short[index:(index + (window_size - 1))])))
end

WAV.wavwrite(moutput, "tm_guitar.wav", Fs=48000)
