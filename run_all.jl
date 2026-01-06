# Main script to run all 4 example files
# Each example is run sequentially

# Global configuration for number of MCMC iterations
const N_ITERATIONS = 100_000

println("="^60)
println("Running all Bayesian Symbolic Regression examples")
println("="^60)

examples = ["SIR.jl", "LV.jl", "Bioreactor.jl", "BioreactorRandom.jl"]

for example in examples
    println("\n" * "="^60)
    println("Running: $example")
    println("="^60 * "\n")
    
    include(example)
    
    println("\n" * "-"^60)
    println("Completed: $example")
    println("-"^60)
end

println("\n" * "="^60)
println("All examples completed!")
println("="^60)
