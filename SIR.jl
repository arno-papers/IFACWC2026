include("common.jl")

# Standard SIR Model
# dS/dt = -β*S*I/N
# dI/dt = β*S*I/N - γ*I  
# dR/dt = γ*I

rng = StableRNGs.StableRNG(2222)

# True parameters
const β0 = 0.3      # transmission rate (R0 ≈ 3)
const γ = 0.1       # recovery rate (≈10 day infectious period)
const N = 1.0       # normalized population

function SIR_true(du, u, p, t)
    S, I, R = u
    du[1] = -β0 * S * I / N
    du[2] = β0 * S * I / N - γ * I
    du[3] = γ * I
end

# Initial conditions
u0 = [0.99, 0.01, 0.0]  # 1% initial infection
tspan = (0.0, 30.0)
saveat = 1.0

prob = ODEProblem(SIR_true, u0, tspan)
sol = solve(prob, Vern7(), saveat=saveat, abstol=1e-12, reltol=1e-12)
y_true = Array(sol) .+ 0.002randn(rng, size(sol))

# Neural network to learn the shared infection term (β*S*I)
rbf(x) = exp.(-(x .^ 2))
const nn = Lux.Chain(Lux.Dense(3, 8, rbf), Lux.Dense(8, 8, rbf), Lux.Dense(8, 8, rbf),
    Lux.Dense(8, 1))  # Single output: the shared interaction term
params, state = Lux.setup(rng, nn)

# UDE: NN outputs shared term β*S*I, used with opposite signs
function SIR_UDE(du, u, p, t)
    S, I, R = u
    interaction = nn(u, p, state)[1][1]  # Single shared term (should learn β*S*I)
    du[1] = -interaction              # dS/dt = -β*S*I
    du[2] = interaction - γ * I       # dI/dt = +β*S*I - γ*I
    du[3] = γ * I                     # Known recovery dynamics
end

prob_ude = ODEProblem(SIR_UDE, u0, tspan, params)

function loss(p, y_true)
    sol = solve(remake(prob_ude, p=p), Vern7(), saveat=saveat,
        sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true)),
        abstol=1e-12, reltol=1e-12)
    if !SciMLBase.successful_retcode(sol)
        return Inf
    end
    y_pred = Array(sol)
    val = sum((y_true .- y_pred) .^ 2)
    println(val)
    val
end

if isfile(joinpath(@__DIR__, "nn_params_sir.jld2"))
    @load joinpath(@__DIR__, "nn_params_sir.jld2") nn_data
    trained_params = ComponentVector{Float64}(params)
    trained_params .= nn_data
else
    optf = OptimizationFunction(loss, AutoZygote())
    optprob = OptimizationProblem(optf, ComponentVector{Float64}(params), y_true)
    optres = solve(optprob, OptimizationOptimisers.Adam(), maxiters=1000)
    optprob2 = OptimizationProblem(optf, optres.u, y_true)
    optres2 = solve(optprob2, Optimization.LBFGS())
    trained_params = optres2.u
    nn_data = Vector(trained_params)
    @save joinpath(@__DIR__, "nn_params_sir.jld2") nn_data
end

# Extract NN predictions for symbolic regression
# Use only S and I as inputs (R is redundant since S + I + R = 1)
X_sr = y_true[1:2, :]
# NN now outputs single value: the shared interaction term β*S*I
Y_sr = Float64[nn(y_true[:, i], trained_params, state)[1][1] for i in 1:size(y_true, 2)]

# True target: β₀*S*I
Y_true_sr = Float64[β0 * X_sr[1, i] * X_sr[2, i] / N for i in 1:size(X_sr, 2)]

n_features = size(X_sr, 1)  # 2 features: S and I
variable_names = ["S", "I"]
# CRITICAL: InverseGamma(1, 1) was causing issues - the β=1 dominated the Gibbs posterior
# making all trees have similar σsq (~0.065) regardless of fit quality!
# Using a much smaller β allows the SSR to dominate the posterior.
σsq_prior = InverseGamma(1, 0.001)
# Broader priors matching Bioreactor.jl for generality
a_prior = Normal(1, 2)
b_prior = Normal(0, 2)
depth_prior_func(d) = Bernoulli(0.9 * (1 + d)^(-0.7))
# Full operator set matching Bioreactor.jl
unary_ops = (neg, exp)
binary_ops = (+, *, /)
ternary_ops = (lt,)
ops = (unary_ops..., binary_ops..., ternary_ops...)
n_ops = length(ops)
ops_prior = Categorical(fill(1 / n_ops, n_ops))
operators = OperatorEnum(1 => unary_ops, 2 => binary_ops, 3 => ternary_ops)

features_prior = Categorical(fill(1 / n_features, n_features))
priors = BSRPriors(depth_prior_func, features_prior, ops_prior, operators,
    σsq_prior, a_prior, b_prior, variable_names)

# Learn shared infection term: should approximate β*S*I
# Run with proper iteration count for convergence
n_trees = 5
trees = []
log_liks = []
for chain in 1:n_trees
    println("=== CHAIN $chain ===")
    tree, σsq, log_lik = run_bayesian_sr(rng, X_sr, Y_sr, priors, n_iter=(@isdefined(N_ITERATIONS) ? N_ITERATIONS : 100_000))
    println(tree)
    push!(trees, tree)
    push!(log_liks, log_lik)
end
# Solve UDE with trained NN parameters
sol_ude = solve(remake(prob_ude, p=trained_params), Vern7(), saveat=saveat, abstol=1e-12, reltol=1e-12)

# Plotting - Two panel layout for clarity
# Panel 1: True vs NN (larger, clearer comparison)
p1 = plot(sol.t, Array(sol)[1, :], label="True S", xlabel="t (days)", ylabel="Population fraction",
    linestyle=:solid, linewidth=3, c=:blue, title="True vs NN Prediction", legend=:right)
plot!(p1, sol.t, Array(sol)[2, :], label="True I", linestyle=:solid, linewidth=3, c=:red)
plot!(p1, sol.t, Array(sol)[3, :], label="True R", linestyle=:solid, linewidth=3, c=:green)
scatter!(p1, sol.t, y_true[1, :], markersize=4, label="Measured S", c=:blue, alpha=0.6, markerstrokewidth=0)
scatter!(p1, sol.t, y_true[2, :], markersize=4, label="Measured I", c=:red, alpha=0.6, markerstrokewidth=0)
scatter!(p1, sol.t, y_true[3, :], markersize=4, label="Measured R", c=:green, alpha=0.6, markerstrokewidth=0)
# Panel 2: SR predictions
p2 = plot(sol.t, Array(sol)[1, :], label="True S", xlabel="t (days)", ylabel="Population fraction",
    linestyle=:solid, linewidth=2, c=:blue, title="SR Predictions", legend=:right, alpha=0.5)
plot!(p2, sol.t, Array(sol)[2, :], label="True I", linestyle=:solid, linewidth=2, c=:red, alpha=0.5)
plot!(p2, sol.t, Array(sol)[3, :], label="True R", linestyle=:solid, linewidth=2, c=:green, alpha=0.5)

# SR predictions using shared tree (negated for dS/dt, positive for dI/dt)
for (i, tree) in enumerate(trees)
    function SIR_SR_i(du, u, p, t)
        S, I, R = u
        # Evaluate shared tree - represents +β*S*I
        sr_pred = eval_tree_array(tree, reshape(u[1:2], 2, 1), operators)[1][1]
        du[1] = -sr_pred           # dS/dt = -β*S*I
        du[2] = sr_pred - γ * I    # dI/dt = +β*S*I - γ*I
        du[3] = γ * I
    end
    prob_sr = ODEProblem(SIR_SR_i, u0, tspan)
    sol_sr = solve(prob_sr, Vern7(), saveat=saveat, abstol=1e-12, reltol=1e-12)
    if SciMLBase.successful_retcode(sol_sr)
        label1 = i == 1 ? "SR S" : ""
        label2 = i == 1 ? "SR I" : ""
        label3 = i == 1 ? "SR R" : ""
        plot!(p2, sol_sr.t, sol_sr[1, :], label=label1, linestyle=:dot, linewidth=1.5, c=:blue, alpha=0.7)
        plot!(p2, sol_sr.t, sol_sr[2, :], label=label2, linestyle=:dot, linewidth=1.5, c=:red, alpha=0.7)
        plot!(p2, sol_sr.t, sol_sr[3, :], label=label3, linestyle=:dot, linewidth=1.5, c=:green, alpha=0.7)
    end
end

# Combine into 2-panel figure for states
p = plot(p1, p2, layout=(2, 1), size=(800, 600))
savefig(p, "fig_sir.pdf")

# Plot of the MISSING PHYSICS: comparing true β*S*I, NN output, and SR predictions
S_vals = X_sr[1, :]
I_vals = X_sr[2, :]
nn_output = Y_sr  # Single NN output (should be β*S*I)
true_SI = β0 .* S_vals .* I_vals

p_physics = plot(sol.t, true_SI, label="True β₀·S·I", linewidth=3, c=:blue,
    xlabel="t (days)", ylabel="Interaction term value", 
    title="Missing Physics: β·S·I", legend=:topright)
plot!(p_physics, sol.t, nn_output, label="NN output", linewidth=2, c=:black, linestyle=:dash)

# Add SR predictions of the missing physics term
for (i, tree) in enumerate(trees)
    sr_pred, complete = eval_tree_array(tree, X_sr, operators)
    if complete
        label = i == 1 ? "SR predictions" : ""
        plot!(p_physics, sol.t, sr_pred, label=label, linestyle=:dot, linewidth=1, c=:red, alpha=0.6)
    end
end

savefig(p_physics, "fig_sir_physics.pdf")
