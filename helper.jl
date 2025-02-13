### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 454d2bb1-0f61-4b7f-b101-d3d32aaf27ec
# Define the training function using Optim.jl
function optim_train(Input, labels, objective; lr=0.01, epochs=500, kwargs...)
    n_features = size(Input, 2)
    params = zeros(n_features) .* 0.0001  # Initialize weights

    # Define the objective function
    objective_function(params) = objective(Input, labels, params; kwargs...)

    # Define the gradient function using Zygote
    function gradient!(storage, params)
        grad = Zygote.gradient(objective_function, params)[1]
        copyto!(storage, grad)  # Copy the gradient to the storage array
    end

    # Use Optim.jl to optimize the parameters
    result = optimize(
        objective_function, 
        gradient!, 
        params, 
        LBFGS(), 
        Optim.Options(
            iterations=epochs, 
            show_trace=true, 
            show_every=100
        )
    )

    # Extract the optimized parameters
    optimized_params = result.minimizer

    println("Training complete.")
    println("Final objective value = $(objective_function(optimized_params))")
    return optimized_params
end;

# ╔═╡ f65d5613-732c-4f0f-b858-aef25db7d3f5
function grid_prediction(range, basis, predictor, args...)
	xy_point = hcat(
		repeat(range, inner=length(range)),
		repeat(range, outer=length(range))
	)
	X_points = [ones(size(xy_point, 1)) xy_point]
	if basis != nothing
		X_points = basis(length_scale, X_points, X)
	end
	cdf_value = predictor(X_points, args...)
	cdf_mesh = reshape(cdf_value, 100, 100)
	return cdf_mesh
end;

# ╔═╡ Cell order:
# ╠═454d2bb1-0f61-4b7f-b101-d3d32aaf27ec
# ╠═f65d5613-732c-4f0f-b858-aef25db7d3f5
