using Pkg;
Pkg.activate(".");
using Random
using OrdinaryDiffEq
using Lux
using StableRNGs
using Plots
using DynamicExpressions
using Distributions
using UnPack
using OrdinaryDiffEq
using Lux
using Optimization
using StableRNGs
using ComponentArrays
using OptimizationOptimisers
using Zygote
using SciMLSensitivity
using JLD2
using Turing
using AdvancedMH: RWMH
using Turing: externalsampler
using DynamicPPL: InitFromParams
using ForwardDiff
using LinearAlgebra: I

neg(x::Float64) = -x
@declare_expression_operator(neg, 1)

lt(x::Float64, a::Float64, b::Float64) = a * x + b
@declare_expression_operator(lt, 3)

struct BSRPriors{D,F,O,Op,S,A,B}
    depth_prior::D
    features_prior::F
    ops_prior::O
    operators::Op
    σsq_prior::S
    a_prior::A
    b_prior::B
    variable_names::Vector{String}
end


function sample_tree_prior(rng::AbstractRNG, depth::Int, priors::BSRPriors)
    is_branch = rand(rng, priors.depth_prior(depth))
    if !is_branch
        feature_index = rand(rng, priors.features_prior)
        return Node{Float64,3}(feature=feature_index)
    else
        op_index = rand(rng, priors.ops_prior)
        n_unary = length(priors.operators.ops[1])
        n_binary = length(priors.operators.ops[2])
        isunary = op_index <= n_unary
        isbinary = n_unary < op_index <= n_unary + n_binary

        if isunary
            return Node{Float64,3}(op=op_index, children=(sample_tree_prior(rng, depth + 1, priors),))
        elseif isbinary
            return Node{Float64,3}(op=op_index - n_unary,
                children=(sample_tree_prior(rng, depth + 1, priors),
                    sample_tree_prior(rng, depth + 1, priors)))
        else # ternary (lt)
            a = rand(rng, priors.a_prior)
            b = rand(rng, priors.b_prior)
            return Node{Float64,3}(op=1,
                children=(sample_tree_prior(rng, depth + 1, priors),
                    Node{Float64,3}(val=a),
                    Node{Float64,3}(val=b)))
        end
    end
end

function sample_prior(rng::AbstractRNG, priors::BSRPriors)
    σsq = rand(rng, priors.σsq_prior)
    tree = sample_tree_prior(rng, 0, priors)
    (; σsq, tree)
end

function logprobability_tree_prior(tree, depth::Int, priors::BSRPriors)
    if has_operators(tree)
        logprior_remainder = if tree.degree == 1
            logprobability_tree_prior(get_child(tree, 1), depth + 1, priors)
        elseif tree.degree == 2
            logprobability_tree_prior(get_child(tree, 1), depth + 1, priors) +
            logprobability_tree_prior(get_child(tree, 2), depth + 1, priors)
        else # lt (ternary)
            logprobability_tree_prior(get_child(tree, 1), depth + 1, priors) +
            logpdf(priors.a_prior, get_child(tree, 2).val) +
            logpdf(priors.b_prior, get_child(tree, 3).val)
        end
        return logpdf(priors.ops_prior, 1) + logpdf(priors.depth_prior(depth), true) + logprior_remainder
    else
        return logpdf(priors.depth_prior(depth), false) + logpdf(priors.features_prior, tree.feature)
    end
end

function loglikelihood_sr(y::Vector{Float64}, tree, X::Matrix{Float64}, σsq::Float64, operators)
    y_pred, complete = eval_tree_array(tree, X, operators)
    if !complete
        return -Inf
    end
    sum(logpdf.(Normal.(y_pred, sqrt(σsq)), y))
end

function get_feature_nodes_with_depth!(nodes, depths, tree, depth)
    if has_operators(tree)
        for i in 1:tree.degree
            get_feature_nodes_with_depth!(nodes, depths, get_child(tree, i), depth + 1)
        end
    elseif !tree.constant
        push!(nodes, tree)
        push!(depths, depth)
    end
    return nothing
end

function get_ops_nodes_with_depth!(nodes, depths, tree, depth)
    if has_operators(tree)
        push!(nodes, tree)
        push!(depths, depth)
        for i in 1:tree.degree
            get_ops_nodes_with_depth!(nodes, depths, get_child(tree, i), depth + 1)
        end
    end
    return nothing
end

function get_feature_and_ops_nodes_with_depth!(nodes, depths, tree, depth)
    if has_operators(tree)
        push!(nodes, tree)
        push!(depths, depth)
        for i in 1:tree.degree
            get_feature_and_ops_nodes_with_depth!(nodes, depths, get_child(tree, i), depth + 1)
        end
    elseif !tree.constant
        push!(nodes, tree)
        push!(depths, depth)
    end
    return nothing
end

function discrete_step!(rng::AbstractRNG, y::Vector{Float64}, X::Matrix{Float64}, tree, σsq::Float64, priors::BSRPriors)
    old_log_likelihood = loglikelihood_sr(y, tree, X, σsq, priors.operators)
    old_log_prior_probability = logprobability_tree_prior(tree, 0, priors)

    actions = Categorical(fill(1 / 6, 6))
    action = rand(rng, actions)

    n_unary = length(priors.operators.ops[1])
    n_binary = length(priors.operators.ops[2])

    if action == 1 # reassign feature
        feature_nodes = filter(node -> (node.degree == 0 && !node.constant), tree)
        isempty(feature_nodes) && return tree
        selected_node = rand(rng, feature_nodes)
        old_tree = deepcopy(selected_node)
        selected_node.feature = rand(rng, priors.features_prior)
        new_log_likelihood = loglikelihood_sr(y, tree, X, σsq, priors.operators)
        log_α = min(0, new_log_likelihood - old_log_likelihood)

    elseif action == 2 # reassign operation
        ops_nodes = filter(node -> node.degree > 0, tree)
        isempty(ops_nodes) && return tree
        selected_node = rand(rng, ops_nodes)
        old_tree = deepcopy(selected_node)
        if selected_node.degree == 1
            selected_node.op = rand(rng, 1:n_unary)
        elseif selected_node.degree == 2
            selected_node.op = rand(rng, 1:n_binary)
        else
            return tree
        end
        new_log_likelihood = loglikelihood_sr(y, tree, X, σsq, priors.operators)
        log_α = min(0, new_log_likelihood - old_log_likelihood)

    elseif action == 3 # grow tree
        feature_nodes = Node{Float64,3}[]
        depths = Int[]
        get_feature_nodes_with_depth!(feature_nodes, depths, tree, 0)
        isempty(feature_nodes) && return tree
        selected_index = rand(rng, 1:length(feature_nodes))
        selected_node = feature_nodes[selected_index]
        depth = depths[selected_index]
        old_tree = deepcopy(selected_node)
        new_tree = sample_tree_prior(rng, depth, priors)
        while new_tree.degree == 0
            new_tree = sample_tree_prior(rng, depth, priors)
        end
        set_node!(selected_node, new_tree)
        new_log_likelihood = loglikelihood_sr(y, tree, X, σsq, priors.operators)
        new_log_prior_probability = logprobability_tree_prior(tree, 0, priors)
        log_probability_jump_to = -log(length(feature_nodes)) +
                                  logprobability_tree_prior(new_tree, depth, priors) -
                                  logpdf(priors.depth_prior(depth), true)
        new_ops_nodes = filter(node -> node.degree > 0, tree)
        log_probability_jump_from = -log(length(new_ops_nodes)) +
                                    logpdf(priors.features_prior, old_tree.feature)
        log_α = min(0, new_log_likelihood + new_log_prior_probability -
                       old_log_likelihood - old_log_prior_probability -
                       log_probability_jump_to + log_probability_jump_from)

    elseif action == 4 # prune tree
        ops_nodes = Node{Float64,3}[]
        depths = Int[]
        get_ops_nodes_with_depth!(ops_nodes, depths, tree, 0)
        isempty(ops_nodes) && return tree
        selected_index = rand(rng, 1:length(ops_nodes))
        selected_node = ops_nodes[selected_index]
        depth = depths[selected_index]
        old_tree = deepcopy(selected_node)
        new_tree = Node{Float64,3}(feature=rand(rng, priors.features_prior))
        set_node!(selected_node, new_tree)
        new_log_likelihood = loglikelihood_sr(y, tree, X, σsq, priors.operators)
        new_log_prior_probability = logprobability_tree_prior(tree, 0, priors)
        log_probability_jump_to = -log(length(ops_nodes)) +
                                  logpdf(priors.features_prior, new_tree.feature)
        new_feature_nodes = filter(node -> (node.degree == 0 && !node.constant), tree)
        log_probability_jump_from = -log(length(new_feature_nodes)) +
                                    logprobability_tree_prior(old_tree, depth, priors) -
                                    logpdf(priors.depth_prior(depth), true)
        log_α = min(0, new_log_likelihood + new_log_prior_probability -
                       old_log_likelihood - old_log_prior_probability -
                       log_probability_jump_to + log_probability_jump_from)

    elseif action == 5 # collapse operator
        ops_nodes = Node{Float64,3}[]
        depths = Int[]
        get_ops_nodes_with_depth!(ops_nodes, depths, tree, 0)
        isempty(ops_nodes) && return tree
        selected_index = rand(rng, 1:length(ops_nodes))
        selected_node = ops_nodes[selected_index]
        depth = depths[selected_index]
        old_tree = deepcopy(selected_node)
        discarded_child = nothing
        new_tree = if old_tree.degree == 1
            get_child(old_tree, 1)
        elseif old_tree.degree == 2
            children = (get_child(old_tree, 1), get_child(old_tree, 2))
            selected_child = rand(rng, (1, 2))
            discarded_child = selected_child == 1 ? children[2] : children[1]
            children[selected_child]
        else # lt
            get_child(old_tree, 1)
        end
        set_node!(selected_node, new_tree)
        new_log_likelihood = loglikelihood_sr(y, tree, X, σsq, priors.operators)
        new_log_prior_probability = logprobability_tree_prior(tree, 0, priors)
        log_probability_jump_to = -log(length(ops_nodes)) -
                                  (old_tree.degree == 2 ? log(2) : log(1))
        new_ops_and_feature_nodes = filter(node -> node.degree > 0 || !node.constant, tree)
        correction_multiple_children = if old_tree.degree == 1
            0.0
        elseif old_tree.degree == 2
            -log(2) + logprobability_tree_prior(discarded_child, depth + 1, priors)
        else
            logpdf(priors.a_prior, get_child(old_tree, 2).val) +
            logpdf(priors.b_prior, get_child(old_tree, 3).val)
        end
        log_probability_jump_from = -log(length(new_ops_and_feature_nodes)) +
                                    logpdf(priors.ops_prior, 1) + correction_multiple_children
        log_α = min(0, new_log_likelihood + new_log_prior_probability -
                       old_log_likelihood - old_log_prior_probability -
                       log_probability_jump_to + log_probability_jump_from)

    elseif action == 6 # expand operator
        ops_and_feature_nodes = Node{Float64,3}[]
        depths = Int[]
        get_feature_and_ops_nodes_with_depth!(ops_and_feature_nodes, depths, tree, 0)
        isempty(ops_and_feature_nodes) && return tree
        selected_index = rand(rng, 1:length(ops_and_feature_nodes))
        selected_node = ops_and_feature_nodes[selected_index]
        depth = depths[selected_index]
        old_tree = deepcopy(selected_node)
        op_index = rand(rng, priors.ops_prior)
        isunary = op_index <= n_unary
        isbinary = n_unary < op_index <= n_unary + n_binary
        grown_child = nothing
        new_tree = if isunary
            Node{Float64,3}(op=op_index, children=(deepcopy(selected_node),))
        elseif isbinary
            selected_child = rand(rng, (1, 2))
            grown_child = sample_tree_prior(rng, depth + 1, priors)
            children = selected_child == 1 ?
                       (deepcopy(selected_node), grown_child) :
                       (grown_child, deepcopy(selected_node))
            Node{Float64,3}(op=op_index - n_unary, children=children)
        else # lt
            a = rand(rng, priors.a_prior)
            b = rand(rng, priors.b_prior)
            Node{Float64,3}(op=1, children=(deepcopy(selected_node),
                Node{Float64,3}(val=a),
                Node{Float64,3}(val=b)))
        end
        set_node!(selected_node, new_tree)
        new_log_likelihood = loglikelihood_sr(y, tree, X, σsq, priors.operators)
        new_log_prior_probability = logprobability_tree_prior(tree, 0, priors)
        correction_multiple_children = if isunary
            0.0
        elseif isbinary
            -log(2) + logprobability_tree_prior(grown_child, depth + 1, priors)
        else
            logpdf(priors.a_prior, get_child(new_tree, 2).val) +
            logpdf(priors.b_prior, get_child(new_tree, 3).val)
        end
        log_probability_jump_to = -log(length(ops_and_feature_nodes)) +
                                  logpdf(priors.ops_prior, op_index) + correction_multiple_children
        new_ops_nodes = filter(node -> node.degree > 0, tree)
        log_probability_jump_from = -log(length(new_ops_nodes)) - (isbinary ? log(2) : log(1))
        log_α = min(0, new_log_likelihood + new_log_prior_probability -
                       old_log_likelihood - old_log_prior_probability -
                       log_probability_jump_to + log_probability_jump_from)
    end

    u = rand(rng, Uniform(0, 1))
    accepted = log(u) < log_α
    if !accepted
        set_node!(selected_node, old_tree)
    end
    return tree
end

function run_bayesian_sr(rng::AbstractRNG, X::Matrix{Float64}, y::Vector{Float64}, priors::BSRPriors;
    n_iter::Int=100_000)
    state = sample_prior(rng, priors)
    tree = state.tree
    σsq = state.σsq

    for i in 1:n_iter

        tree = discrete_step!(rng, y, X, tree, σsq, priors)
        # MH step for continuous parameters (σsq and a/b if lt nodes exist) every 100 iterations
        i % 1000 != 0 && continue
        a_b_vals_current, a_b_nodes = get_scalar_constants(tree)
        n_lt = length(a_b_nodes) ÷ 2
        initial_params = if n_lt > 0
            pairs = [(:σsq => σsq)]
            for j in 1:2*n_lt
                push!(pairs, Symbol("a_b_vals[$j]") => a_b_vals_current[j])
            end
            NamedTuple(pairs)
        else
            (σsq = σsq,)
        end
        n_params = 1 + 2 * n_lt
        chain = sample(rng, continuous_model(y, X, tree, priors), externalsampler(RWMH(MvNormal(zeros(n_params), 0.1 * I))), 10; progress=false, verbose=false, num_warmup=0, initial_params=InitFromParams(initial_params))
        if n_lt > 0
            a_b_vals = Float64[]
            for j in 1:2:2*n_lt
                push!(a_b_vals, chain[Symbol("a_b_vals[$j]")][end])
                push!(a_b_vals, chain[Symbol("a_b_vals[$(j+1)]")][end])
            end
            set_scalar_constants!(tree, a_b_vals, a_b_nodes)
        end
        σsq = chain[:σsq][end]
    end

    # Final NUTS step with 100 iterations (no warmup)
    a_b_vals_current, a_b_nodes = get_scalar_constants(tree)
    n_lt = length(a_b_nodes) ÷ 2
    final_initial_params = if n_lt > 0
        pairs = [(:σsq => σsq)]
        for j in 1:2*n_lt
            push!(pairs, Symbol("a_b_vals[$j]") => a_b_vals_current[j])
        end
        NamedTuple(pairs)
    else
        (σsq = σsq,)
    end
    chain = sample(rng, continuous_model(y, X, tree, priors), NUTS(), 1_000; progress=false, verbose=false, num_warmup=0, initial_params=InitFromParams(final_initial_params))
    _, a_b_nodes = get_scalar_constants(tree)
    n_lt = length(a_b_nodes) ÷ 2
    if n_lt > 0
        a_b_vals = Float64[]
        for j in 1:2:2*n_lt
            push!(a_b_vals, chain[Symbol("a_b_vals[$j]")][end])
            push!(a_b_vals, chain[Symbol("a_b_vals[$(j+1)]")][end])
        end
        set_scalar_constants!(tree, a_b_vals, a_b_nodes)
    end
    σsq = chain[:σsq][end]

    log_lik = loglikelihood_sr(y, tree, X, σsq, priors.operators)
    return tree, σsq, log_lik
end

@model function continuous_model(y, X, tree, priors)
    @unpack σsq_prior, a_prior, b_prior, operators = priors
    _, a_b_nodes = get_scalar_constants(tree)
    n_lt = length(a_b_nodes) ÷ 2
    σsq ~ σsq_prior
    a_b_vals = Vector{Float64}(undef, 2 * n_lt)
    for i in 1:2:2*n_lt
        a_b_vals[i] ~ a_prior
        a_b_vals[i+1] ~ b_prior
    end
    y_sim = if length(a_b_vals) == 0
        eval_tree_array(tree, X, operators)[1]
    elseif eltype(a_b_vals) == Float64
        set_scalar_constants!(tree, a_b_vals, a_b_nodes)
        eval_tree_array(tree, X, operators)[1]
    else
        set_scalar_constants!(tree, ForwardDiff.value.(a_b_vals), a_b_nodes)
        val, grad = eval_grad_tree_array(tree, X, operators, variable=false)
        [ForwardDiff.Dual{ForwardDiff.tagtype(a_b_vals[1])}(val[i], 0.0, grad[:, i]...) for i in eachindex(val)]
    end
    for i in 1:length(y)
        y[i] ~ Normal(y_sim[i], sqrt(σsq))
    end
    return nothing
end

function run_continuous_sampling(rng::AbstractRNG, X::Matrix{Float64}, y::Vector{Float64}, tree, priors::BSRPriors;
    n_samples::Int=1000)
    tree_copy = copy(tree)
    _, a_b_nodes = get_scalar_constants(tree_copy)
    n_lt = length(a_b_nodes) ÷ 2
    n_params = 1 + 2 * n_lt
    chain = sample(rng, continuous_model(y, X, tree_copy, priors), externalsampler(RWMH(MvNormal(zeros(n_params), 0.1 * I))), n_samples)
    return chain, tree_copy
end
