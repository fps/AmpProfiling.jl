import WAV

training_input() = WAV.wavread("guitar_clean.wav")[1]

training_output() = WAV.wavread("guitar_processed.wav")[1]

test_input() = WAV.wavread("guitar_short.wav")[1][:,1]

function unroll_time(input, window_size)
    N = length(input) - window_size
    output = Array{Any}(N,1)

    for index in 1:N
        output[index] = view(input, index:(index+window_size-1))
    end
    return output
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

