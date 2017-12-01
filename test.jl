include("AmpProfiling.jl")
import WAV

guitar_clean = WAV.wavread("guitar_clean.wav")
guitar_processed = WAV.wavread("guitar_processed.wav")
guitar_short = WAV.wavread("guitar_short.wav")

window_size = 32

m = AmpProfiling.create_model(window_size)

tm = AmpProfiling.train_model(m, guitar_clean[1][:,1], guitar_processed[1][:,1], 32, 1000)

moutput = (Float64)[]
for index in 1:(size(guitar_short[1], 1) - window_size)
    append!(moutput, Flux.Tracker.value(tm(guitar_short[1][index:(index + (window_size - 1))])))
end

WAV.wavwrite(moutput, "tm_guitar.wav", Fs=48000)
