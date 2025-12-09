include("common.jl")


rng = StableRNGs.StableRNG(1111)

# True parameters
const μ_max = 0.421    # Maximum specific growth rate (1/h)
const K_s = 4.39       # Monod constant (g/L)
const C_s_in = 50.0    # Inlet substrate concentration (g/L)
const y_x_s = 0.777    # Yield coefficient

# Extended flow profile Q_in(t) - 45 hours total
const Q_in_profile = [
    (0.0, 0.0),    # t=0-15h: Q_in = 0
    (15.0, 0.2),   # t=15-18h: Q_in = 0.2
    (18.0, 0.5),   # t=18-21h: Q_in = 0.5
    (21.0, 0.8),   # t=21-24h: Q_in = 0.8
    (24.0, 0.3),   # t=24-27h: Q_in = 0.3
    (27.0, 0.6),   # t=27-30h: Q_in = 0.6
    (30.0, 1.0),   # t=30-45h: Q_in = 1.0
]

@inline function get_Q_in(t)
    t < 15.0 && return 0.0
    t < 18.0 && return 2.0
    t < 21.0 && return 0.0
    t < 24.0 && return 1.0
    t < 27.0 && return 3.0
    t < 30.0 && return 7.0
    return 10.0
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

sol1 = solve(prob1, Rodas5(), saveat=saveat, abstol=1e-10, reltol=1e-10)
sol2 = solve(prob2, Rodas5(), saveat=saveat, abstol=1e-10, reltol=1e-10, tstops=[18.0, 21.0, 24.0, 27.0])
sol3 = solve(prob3, Rodas5(), saveat=saveat, abstol=1e-10, reltol=1e-10)

t_all = vcat(sol1.t, sol2.t, sol3.t)
y_all = hcat(Array(sol1), Array(sol2), Array(sol3))
y_true = y_all .+ 0.1randn(rng, size(y_all))
plot(t_all, y_all', label = ["Cs optimal experiment" "Cx optimal experiment" "V optimal experiment"], xlabel="t", linestyle=:solid, linewidth=2)
scatter!(t_all, y_true[1,:], label = "Cs optimal experiment measurements", markersize=3, c=1)

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
plot!(t_all, y_all', label = ["Cs random experiment" "Cx random experiment" "V random experiment"], xlabel="t", linestyle=:dash, linewidth=2)
scatter!(t_all, y_true[1,:], label = "Cs random experiment measurements", markersize=3, c=1, m=:square)
savefig("fig_bioreactor_states.pdf")
