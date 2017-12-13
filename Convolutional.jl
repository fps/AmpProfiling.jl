module ConvolutionalLayer
    import Flux
    import Flux.Tracker

    struct Convolutional
        W
        b
        N
        in
        out
    end

    Convolutional(in::Integer, out::Integer, N::Integer) =
        Convolutional(Flux.Tracker.param(randn(out, in)), Flux.Tracker.param(randn(out)), N, in, out)

    # Overload call, so the object can be used as a function
    function (m::Convolutional)(x)
        x_reshaped = reshape(x, m.in, m.N)
        return reshape(m.W * x_reshaped .+ m.b, m.N*m.out, 1)
    end
end
