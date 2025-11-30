using OrdinaryDiffEq
using Lux
using Optimization # needs to be v4 not v5
using StableRNGs
using ComponentArrays
using OptimizationOptimisers
using Zygote
using SciMLSensitivity
using Plots

rng = StableRNGs.StableRNG(1111)

function LV(du, u, p, t)
    du[1] = 1.3 * u[1] - 0.9*u[1] * u[2]
    du[2] = -1.8 * u[2] + 0.8*u[1] * u[2]
end
u0 = [3.1461493970111687, 1.5370475785612603]
tspan = (0.0, 5.0)
prob = ODEProblem(LV, u0, tspan)
sol = solve(prob, Vern7(), saveat=0.25, abstol=1e-12, reltol=1e-12,)
y_true = Array(sol)
plot(sol.t, y_true')

rbf(x) = exp.(-(x .^ 2))
const U = Lux.Chain(Lux.Dense(2, 5, rbf), Lux.Dense(5, 5, rbf), Lux.Dense(5, 5, rbf),
    Lux.Dense(5, 2))
p, st = Lux.setup(rng, U)
const _st = st

function LV_UDE(du, u, p, t)
    NN = U(u, p, _st)[1]
    du[1] = 1.3 * u[1] + NN[1]
    du[2] = -1.8 * u[2] + NN[2]
end

prob_ude = ODEProblem(LV_UDE, u0, tspan, p)


function loss(p, y_true)
    sol = solve(remake(prob_ude, p=p), Vern7(), saveat=0.25,
        sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true)),
        abstol=1e-12, reltol=1e-12,)
    y_pred = Array(sol)
    val = sum((y_true .- y_pred) .^ 2)
    println(val)
    val
end
optf = OptimizationFunction(loss, AutoZygote())
optprob = OptimizationProblem(optf, ComponentVector{Float64}(p), y_true)
optres = solve(optprob, OptimizationOptimisers.Adam(), maxiters=1000)
optprob2 = OptimizationProblem(optf, optres.u, y_true)
optres2 = solve(optprob2, Optimization.LBFGS())

sol_test = solve(remake(prob_ude, p=optres2.u), Tsit5(), saveat=0:0.1:10)
plot(sol_test)
