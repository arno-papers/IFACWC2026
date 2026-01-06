include("common.jl")


rng = StableRNGs.StableRNG(3333)

# True parameters
const μ_max = 0.421    # Maximum specific growth rate (1/h)
const K_s = 4.39       # Monod constant (g/L)
const C_s_in = 50.0    # Inlet substrate concentration (g/L)
const y_x_s = 0.777    # Yield coefficient

# Generate random Q_in values at each 3-hour time segment (15 segments total for 45 hours)
Q_in_values = 5.0 * rand(rng, 15)  # Random values between 0 and 10

@inline function get_Q_in(t)
    idx = min(Int(floor(t / 3.0)) + 1, 15)
    return Q_in_values[idx]
end

function bioreactor_true!(du, u, p, t)
    C_s, C_x, V = u
    Q_in = get_Q_in(t)
    
    μ = 0.421 * C_s / (4.39 + C_s)
    σ = μ / 0.777
    
    du[1] = -σ * C_x + Q_in / V * (50.0 - C_s)
    du[2] = μ * C_x - Q_in / V * C_x
    du[3] = Q_in
end

u0 = [10.0, 1.0, 7.0]
saveat = 0.5

# Solve in 3 segments of 15 hours each
prob1 = ODEProblem(bioreactor_true!, u0, (0.0, 15.0))
prob2 = ODEProblem(bioreactor_true!, u0, (15.0, 30.0))
prob3 = ODEProblem(bioreactor_true!, u0, (30.0, 45.0))

sol1 = solve(prob1, Rodas5(), saveat=saveat, abstol=1e-10, reltol=1e-10, tstops=[3.0, 6.0, 9.0, 12.0])
sol2 = solve(prob2, Rodas5(), saveat=saveat, abstol=1e-10, reltol=1e-10, tstops=[18.0, 21.0, 24.0, 27.0])
sol3 = solve(prob3, Rodas5(), saveat=saveat, abstol=1e-10, reltol=1e-10, tstops=[33.0, 36.0, 39.0, 42.0])

t_all = vcat(sol1.t, sol2.t, sol3.t)
y_all = hcat(Array(sol1), Array(sol2), Array(sol3))
y_true = y_all .+ 0.1randn(rng, size(y_all))

sigmoid(x) = 1 / (1 + exp(-x))
rbf(x) = exp.(-(x .^ 2))
const nn = Lux.Chain(
    Lux.Dense(1, 5, tanh),
    Lux.Dense(5, 5, tanh),
    Lux.Dense(5, 1, x -> sigmoid.(x))
)
params, state = Lux.setup(rng, nn)

function bioreactor_ude!(du, u, p, t)
    C_s, C_x, V = u
    Q_in = get_Q_in(t)
    
    μ = nn([C_s], p, state)[1][1]
    σ = μ / 0.777
    
    du[1] = -σ * C_x + Q_in / V * (50.0 - C_s)
    du[2] = μ * C_x - Q_in / V * C_x
    du[3] = Q_in
end

prob_ude1 = ODEProblem(bioreactor_ude!, u0, (0.0, 15.0), params)
prob_ude2 = ODEProblem(bioreactor_ude!, u0, (15.0, 30.0), params)
prob_ude3 = ODEProblem(bioreactor_ude!, u0, (30.0, 45.0), params)

y_true1 = y_true[:, 1:length(sol1.t)]
y_true2 = y_true[:, length(sol1.t)+1:length(sol1.t)+length(sol2.t)]
y_true3 = y_true[:, length(sol1.t)+length(sol2.t)+1:end]

function loss(p, _)
    s1 = solve(remake(prob_ude1, p=p), AutoTsit5(Rodas5()), saveat=saveat,
        sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true)),
        abstol=1e-8, reltol=1e-8, tstops=[3.0, 6.0, 9.0, 12.0])
    s2 = solve(remake(prob_ude2, p=p), AutoTsit5(Rodas5()), saveat=saveat,
        sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true)),
        abstol=1e-8, reltol=1e-8, tstops=[18.0, 21.0, 24.0, 27.0])
    s3 = solve(remake(prob_ude3, p=p), AutoTsit5(Rodas5()), saveat=saveat,
        sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true)),
        abstol=1e-8, reltol=1e-8, tstops=[33.0, 36.0, 39.0, 42.0])
    if !SciMLBase.successful_retcode(s1) || !SciMLBase.successful_retcode(s2) || !SciMLBase.successful_retcode(s3)
        return Inf
    end
    val = sum((y_true1 .- Array(s1)) .^ 2) + sum((y_true2 .- Array(s2)) .^ 2) + sum((y_true3 .- Array(s3)) .^ 2)
    println(val)
    val
end

if isfile(joinpath(@__DIR__, "nn_params_bioreactor_random.jld2"))
    @load joinpath(@__DIR__, "nn_params_bioreactor_random.jld2") nn_data
    trained_params = ComponentVector{Float64}(params)
    trained_params .= nn_data
else
    optf = OptimizationFunction(loss, AutoZygote())
    optprob = OptimizationProblem(optf, ComponentVector{Float64}(params), y_true)
    optres = solve(optprob, OptimizationOptimisers.Adam(0.2), maxiters=3000)
    optprob = OptimizationProblem(optf, optres.u, y_true)
    optres = solve(optprob, OptimizationOptimisers.Adam(0.01), maxiters=3000)
    optprob2 = OptimizationProblem(optf, optres.u, y_true)
    optres2 = solve(optprob2, Optimization.LBFGS())
    trained_params = optres2.u
    nn_data = Vector(trained_params)
    @save joinpath(@__DIR__, "nn_params_bioreactor_random.jld2") nn_data
end

X_sr = y_true[1:1, :]
Y_sr = hcat([nn([X_sr[1, i]], trained_params, state)[1] for i in 1:size(X_sr, 2)]...)'
Y_true = hcat([μ_max * X_sr[1, i] / (K_s + X_sr[1, i]) for i in 1:size(X_sr, 2)]...)'
n_features = size(X_sr, 1)
variable_names = ["Cs"]
# Fixed: InverseGamma(1,1) β=1 dominated Gibbs posterior
σsq_prior = InverseGamma(1, 0.001)
a_prior = Normal(1, 2)
b_prior = Normal(0, 2)
depth_prior_func(d) = Bernoulli(0.9 * (1 + d)^(-0.7))
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

y1 = collect(vec(Y_sr))
trees1 = []
log_liks1 = []
for chain in 1:100
    println(chain)
    tree, σsq, log_lik = run_bayesian_sr(rng, X_sr, y1, priors, n_iter=(@isdefined(N_ITERATIONS) ? N_ITERATIONS : 100_000))
    println(tree)
    push!(trees1, tree)
    push!(log_liks1, log_lik)
end

# Sort by log likelihood and take top 10 unique equations (up to 1e-5 difference in likelihood)
sorted_indices = sortperm(log_liks1, rev=true)
top_indices = Int[]
for idx in sorted_indices
    is_unique = true
    for top_idx in top_indices
        if abs(log_liks1[idx] - log_liks1[top_idx]) < 1e-5
            is_unique = false
            break
        end
    end
    if is_unique
        push!(top_indices, idx)
    end
    length(top_indices) >= 10 && break
end
top_trees = trees1[top_indices]

# Print top 10 equations
println("\nTop 10 unique equations by log likelihood:")
for (i, idx) in enumerate(top_indices)
    println("$i. (log_lik=$(log_liks1[idx])): $(trees1[idx])")
end

Cs_range = range(0, 50, length=200)
Y_monod = [μ_max * Cs / (K_s + Cs) for Cs in Cs_range]
p1 = plot(Cs_range, Y_monod, label="True μ (Monod)", xlabel="Cs", ylabel="μ (growth rate)", linestyle=:solid, linewidth=2, c=1)
scatter!(p1, X_sr[1, :], vec(Y_sr), label="NN prediction", markersize=3, c=2, ylims=(0,0.5))

# Add SR predictions from top 10 trees
Cs_sorted = sort(vec(X_sr[1, :]))
for (i, tree) in enumerate(top_trees)
    Y_sr_pred, _ = eval_tree_array(tree, reshape(Cs_sorted, 1, :), operators)
    label = i == 1 ? "SR predictions (top 10)" : ""
    plot!(p1, Cs_sorted, Y_sr_pred, label=label, linestyle=:dot, linewidth=1, c=3, alpha=0.6)
end

savefig(p1, "fig_bioreactor_random.pdf")
