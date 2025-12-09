include("common.jl")


rng = StableRNGs.StableRNG(1111)

function LV(du, u, p, t)
    du[1] = 1.3 * u[1] - 0.9 * u[1] * u[2]
    du[2] = -1.8 * u[2] + 0.8 * u[1] * u[2]
end

u0 = [3.1461493970111687, 1.5370475785612603]
tspan = (0.0, 5.0)
saveat = 0.25

prob = ODEProblem(LV, u0, tspan)
sol = solve(prob, Vern7(), saveat=saveat, abstol=1e-12, reltol=1e-12)
y_true = Array(sol) .+ 0.25randn(rng, size(sol))
rbf(x) = exp.(-(x .^ 2))
const nn = Lux.Chain(Lux.Dense(2, 5, rbf), Lux.Dense(5, 5, rbf), Lux.Dense(5, 5, rbf),
    Lux.Dense(5, 2))
params, state = Lux.setup(rng, nn)

function LV_UDE(du, u, p, t)
    NN = nn(u, p, state)[1]
    du[1] = 1.3 * u[1] + NN[1]   # Missing: -0.9*u1*u2
    du[2] = -1.8 * u[2] + NN[2]  # Missing: +0.8*u1*u2
end

prob_ude = ODEProblem(LV_UDE, u0, tspan, params)
function loss(p, y_true)
    sol = solve(remake(prob_ude, p=p), Vern7(), saveat=saveat,
        sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true)),
        abstol=1e-12, reltol=1e-12)
    y_pred = Array(sol)
    val = sum((y_true .- y_pred) .^ 2)
    println(val)
    val
end

if isfile(joinpath(@__DIR__, "nn_params_lv.jld2"))
    @load joinpath(@__DIR__, "nn_params_lv.jld2") nn_data
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
    @save joinpath(@__DIR__, "nn_params_lv.jld2") nn_data
end

X_sr = y_true
Y_sr = hcat([nn(y_true[:, i], trained_params, state)[1] for i in 1:size(X_sr, 2)]...)
Y_true = hcat([[-0.9 * X_sr[1, i] * X_sr[2, i], 0.8 * X_sr[1, i] * X_sr[2, i]] for i in 1:size(X_sr, 2)]...)
n_features = size(X_sr, 1)
variable_names = ["x$i" for i in 1:n_features]
σsq_prior = InverseGamma(1, 1)
a_prior = Normal(1, 1)
b_prior = Normal(0, 1)
depth_prior_func(d) = Bernoulli(0.9 * (1 + d)^(-0.7))
unary_ops = (neg, exp)
binary_ops = (+, *)
ternary_ops = (lt,)
ops = (unary_ops..., binary_ops..., ternary_ops...)
n_ops = length(ops)
ops_prior = Categorical(fill(1 / n_ops, n_ops))
operators = OperatorEnum(1 => unary_ops, 2 => binary_ops, 3 => ternary_ops)

features_prior = Categorical(fill(1 / n_features, n_features))
priors = BSRPriors(depth_prior_func, features_prior, ops_prior, operators,
    σsq_prior, a_prior, b_prior, variable_names)
n_trees = 10

y1 = Y_sr[1, :]
trees1 = []
for chain in 1:n_trees
    tree, σsq, log_lik = run_bayesian_sr(rng, X_sr, y1, priors, n_iter=1_000_000)
    println(tree)
    push!(trees1, tree)
end
y2 = Y_sr[2, :]
trees2 = []
for chain in 1:n_trees
    tree, σsq, log_lik = run_bayesian_sr(rng, X_sr, y2, priors, n_iter=1_000_000)
    println(tree)
    push!(trees2, tree)
end

# Solve UDE with trained NN parameters
sol_ude = solve(remake(prob_ude, p=trained_params), Vern7(), saveat=saveat, abstol=1e-12, reltol=1e-12)

p1 = plot(sol.t, Array(sol)[1, :], label="True x1 (prey)", xlabel="t", ylabel="Population size", linestyle=:solid, linewidth=1.5, c=1, ylim = (0,4))
plot!(p1, sol.t, Array(sol)[2, :], label="True x2 (predator)", linestyle=:solid, linewidth=1.5, c=2)
scatter!(p1, sol.t, X_sr[1, :], markersize=3, label="Measured x1", c=1)
scatter!(p1, sol.t, X_sr[2, :], markersize=3, label="Measured x2", c=2)
# NN prediction of states
plot!(p1, sol_ude.t, sol_ude[1, :], label="NN x1", linestyle=:dash, linewidth=2, c=1)
plot!(p1, sol_ude.t, sol_ude[2, :], label="NN x2", linestyle=:dash, linewidth=2, c=2)

# SR predictions using last 10 trees - separate line for each
for (i, (tree1, tree2)) in enumerate(zip(trees1, trees2))
    function LV_SR_i(du, u, p, t)
        sr1_pred = eval_tree_array(tree1, reshape(u, 2, 1), operators)[1][1]
        sr2_pred = eval_tree_array(tree2, reshape(u, 2, 1), operators)[1][1]
        du[1] = 1.3 * u[1] + sr1_pred
        du[2] = -1.8 * u[2] + sr2_pred
    end
    prob_sr = ODEProblem(LV_SR_i, u0, tspan)
    sol_sr = solve(prob_sr, Vern7(), saveat=saveat, abstol=1e-12, reltol=1e-12)
    label1 = i == 1 ? "SR x1" : ""
    label2 = i == 1 ? "SR x2" : ""
    plot!(p1, sol_sr.t, sol_sr[1, :], label=label1, linestyle=:dot, linewidth=1, c=1, alpha=0.6)
    plot!(p1, sol_sr.t, sol_sr[2, :], label=label2, linestyle=:dot, linewidth=1, c=2, alpha=0.6)
end

savefig(p1, "figLV.pdf")
