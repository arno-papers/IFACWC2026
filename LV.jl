using OrdinaryDiffEqTsit5
using Lux
using Optimization
using StableRNGs
using ComponentArrays
using OptimizationOptimJL
using OptimizationOptimisers
using Zygote
using SciMLSensitivity

rng = StableRNGs.StableRNG(1111)

function LV(du, u, p, t)
    du[1] = 1.5 * u[1] - u[1] * u[2]
    du[2] = -3.0 * u[2] + u[1] * u[2]
end
u0 = [1.0, 1.0]
tspan = (0.0, 10.0)
prob = ODEProblem(LV, u0, tspan)
sol = solve(prob, Tsit5(), saveat=0:0.1:10)
y_true = Array(sol)

rbf(x) = exp.(-(x .^ 2))
const U = Lux.Chain(Lux.Dense(2, 5, rbf), Lux.Dense(5, 5, rbf), Lux.Dense(5, 5, rbf),
    Lux.Dense(5, 2))
p, st = Lux.setup(rng, U)
const _st = st

function LV_UDE(du, u, p, t)
    NN = U(u, p, _st)[1]
    du[1] = 1.5 * u[1] - NN[1]
    du[2] = -3.0 * u[2] + NN[2]
end

prob_ude = ODEProblem(LV_UDE, u0, tspan, p)


function loss(p, y_true)
    sol = solve(remake(prob_ude, p=p), Tsit5(), saveat=0:0.1:10,
        sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true)))
    y_pred = Array(sol)
    val = sum((y_true .- y_pred) .^ 2)
    println(val)
    val
end
optf = OptimizationFunction(loss, AutoZygote())
optprob = OptimizationProblem(optf, ComponentVector{Float64}(p), y_true)
optres = solve(optprob, OptimizationOptimisers.Adam(), maxiters=5000)
optprob2 = OptimizationProblem(optf, optres.u, y_true)
res2 = solve(optprob2, OptimizationOptimJL.BFGS())
