function createCPPCode(model, name)
    includes = """
        #pragma once
        #include <Eigen/Dense>
        #include <vector>
        """

    parameters = """
            std::vector<Eigen::MatrixXd> W;
            std::vector<Eigen::MatrixXd> b;
        """

    constructor_body_parts = Array{String}(length(model.layers), 1)

    for layer in 1:length(model.layers)
        rows = size(model.layers[layer].W, 1)
        cols = size(model.layers[layer].W, 2)

        creation = """
                    W.push_back(Eigen::MatrixXd($(rows), $(cols)));
                    b.push_back(Eigen::MatrixXd($(rows), 1));
            """

        initialization_parts_W = Array{String}(rows,cols)
        initialization_parts_b = Array{String}(rows)
        for row in 1:rows
            initialization_parts_b[row] = """
                        b[$(layer-1)]($(row-1), 0) = $(Flux.Tracker.value(model.layers[layer].b[row]));
                """
            for col in 1:cols
                initialization_parts_W[row, col] = """
                            W[$(layer-1)]($(row-1), $(col-1)) = $(Flux.Tracker.value(model.layers[layer].W[row, col]));
                    """
            end
        end

        constructor_body_parts[layer] = """
                    // Layer $(layer-1)
            $(creation)
            $(string(initialization_parts_W...))
            $(string(initialization_parts_b...))
            """
    end


    constructor = """
            $(name)()
            {
        $(string(constructor_body_parts...))
            }
        """

    struct_body = """
        $(parameters)
        $(constructor)
        """

    cppstruct = """
        struct $(name)
        {
        $(struct_body)
        };
        """

    cpp = """
        $(includes)
        $(cppstruct)
        """
    return cpp

end

