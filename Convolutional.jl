module ConvolutionalLayer
    import Flux
    import Flux.Tracker

    struct Convolutional
        f
        N
        in
        out
    end

    #Convolutional(f, in::Integer, out::Integer, N::Integer) =
    #    Convolutional(f, N, in, out)

    # Overload call, so the object can be used as a function
    function (m::Convolutional)(x)
        x_reshaped = reshape(x, m.in, m.N)
        return reshape(m.f(x_reshaped), m.N*m.out, 1)
    end
end
