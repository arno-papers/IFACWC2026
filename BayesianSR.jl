using DynamicExpressions
using Distributions
using UnPack
using Turing
using Plots
using StatsPlots
using Zygote
using ForwardDiff
lt(x::Float64, a::Float64, b::Float64) = a * x + b # the only way to get non-feature numbers into the equation.
@declare_expression_operator(lt, 3)

σsq_prior = InverseGamma(1, 1)
a_prior = Normal(1, 10)
b_prior = Normal(0, 10)

unary_ops = (-, inv, exp)
binary_ops = (+, *)
ternary_ops = (lt,)
ops = (unary_ops..., binary_ops..., ternary_ops...)
n_ops = length(ops)
ops_prior = Categorical(fill(1 / n_ops, n_ops))

operators = OperatorEnum(1 => unary_ops, 2 => binary_ops, 3 => ternary_ops)

n_features = 2
variable_names = ["x$i" for i in 1:n_features]
features_prior = Categorical(fill(1 / n_features, n_features))

α = 0.9
β = 0.7
depth_prior(d, α, β) = Bernoulli(α * (1 + d)^(-β))
depth_prior(d) = depth_prior(d, α, β)

priors = (; depth_prior, features_prior, ops_prior, operators, σsq_prior, a_prior, b_prior,)
function sample_prior(priors)
    @unpack σsq_prior = priors
    σsq = rand(σsq_prior)
    tree = sample_tree_prior(0, priors)
    (; σsq, tree)
end
function sample_tree_prior(depth, priors)
    @unpack depth_prior, features_prior, ops_prior, operators, a_prior, b_prior = priors
    is_branch = rand(depth_prior(depth))
    if !is_branch
        feature_index = rand(features_prior)
        return Node{Float64,3}(feature=feature_index)
    else
        op_index = rand(ops_prior)
        isunary = op_index <= length(operators.ops[1])
        isbinary = length(operators.ops[1]) < op_index <= length(operators.ops[1]) + length(operators.ops[2])
        if isunary
            return Node{Float64,3}(op=op_index, children=(sample_tree_prior(depth + 1, priors),))
        elseif isbinary
            return Node{Float64,3}(op=op_index - length(operators.ops[1]), children=(sample_tree_prior(depth + 1, priors),
                sample_tree_prior(depth + 1, priors)))
        else # assumption lt is only ternary operator
            a = rand(a_prior)
            b = rand(b_prior)
            return Node{Float64,3}(op=1, children=(sample_tree_prior(depth + 1, priors), Node{Float64,3}(val=a), Node{Float64,3}(val=b)))
        end
    end
end

function probability_tree_prior(tree, depth, priors)
    @unpack depth_prior, features_prior, ops_prior, operators, a_prior, b_prior = priors
    if has_operators(tree)
        prior_remainder = if tree.degree == 1
            probability_tree_prior(get_child(tree, 1), depth + 1, priors)
        elseif tree.degree == 2
            probability_tree_prior(get_child(tree, 1), depth + 1, priors) * probability_tree_prior(get_child(tree, 2), depth + 1, priors)
        else # lt
            probability_tree_prior(get_child(tree, 1), depth + 1, priors) * pdf(a_prior, get_child(tree, 2).val) * pdf(b_prior, get_child(tree, 3).val)
        end
        return pdf(ops_prior, 1) * pdf(depth_prior(depth), true) * prior_remainder # TODO generalize 1
    else
        return pdf(depth_prior(depth), false) * pdf(features_prior, tree.feature)
    end
end

function logprobability_tree_prior(tree, depth, priors)
    @unpack depth_prior, features_prior, ops_prior, operators, a_prior, b_prior = priors
    if has_operators(tree)
        logprior_remainder = if tree.degree == 1
            logprobability_tree_prior(get_child(tree, 1), depth + 1, priors)
        elseif tree.degree == 2
            logprobability_tree_prior(get_child(tree, 1), depth + 1, priors) + logprobability_tree_prior(get_child(tree, 2), depth + 1, priors)
        else # lt
            logprobability_tree_prior(get_child(tree, 1), depth + 1, priors) + logpdf(a_prior, get_child(tree, 2).val) + logpdf(b_prior, get_child(tree, 3).val)
        end
        return logpdf(ops_prior, 1) + logpdf(depth_prior(depth), true) + logprior_remainder # TODO generalize 1
    else
        return logpdf(depth_prior(depth), false) + logpdf(features_prior, tree.feature)
    end
end


function loglikelihood(y, tree, X, σsq)
    sum(logpdf.(Normal.(tree(X), sqrt(σsq)), y))
end
testo = sample_prior(priors)
log(probability_tree_prior(testo.tree, 0, priors))
logprobability_tree_prior(testo.tree, 0, priors)

X = rand(2, 100)
y = testo.tree(X)
loglikelihood(y, testo.tree, X, testo.σsq) # -140.08992970719396

@model function continuous(y, X, tree, priors)
    @unpack σsq_prior, a_prior, b_prior, operators = priors
    _, a_b_nodes = get_scalar_constants(tree)
    n_lt = length(a_b_nodes) ÷ 2
    σsq ~ σsq_prior
    a_b_vals = Vector{Float64}(undef, 2 * n_lt)
    for i in 1:2:2*n_lt
        a_b_vals[i] ~ a_prior
        a_b_vals[i+1] ~ b_prior
    end
    println(a_b_vals)
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
continuous_step = sample(continuous(y, X, copy(testo.tree), priors), NUTS(), 1000)

_, a_b_nodes = get_scalar_constants(testo.tree)
n_lt = length(a_b_nodes) ÷ 2

continuous_step = sample(continuous(y, X, copy(testo.tree), priors), MH(), 100_000)
continuous_step = sample(continuous(y, X, copy(testo.tree), priors), MH(I(2 * n_lt + 1)), 100_000, num_warmup=10_000)
describe(continuous_step)
plot(continuous_step)
eval_tree_array(tree, X, operators)
eval_grad_tree_array(tree, X, operators, variable=false)


continuous_step = sample(continuous(y, X, testo.tree, priors), Gibbs(:σsq => MH(I(1)), @varname(a_b_vals[1]) => MH(I(1)), @varname(a_b_vals[2]) => MH(I(1))), 100_000)
describe(continuous_step)
plot(continuous_step)

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

function discrete(y, X, tree, σsq)
    @unpack features_prior, operators, depth_prior, ops_prior, a_prior, b_prior = priors

    old_log_likelihood = loglikelihood(y, tree, X, σsq)
    old_log_prior_probability = logprobability_tree_prior(tree, 0, priors)
    actions = Categorical(fill(1 / 6, 6))
    action = rand(actions)
    if action == 1 # reasign feature
        feature_nodes = filter(node -> (node.degree == 0 && !node.constant), tree)
        selected_node = rand(feature_nodes)
        old_tree = deepcopy(selected_node)
        selected_node.feature = rand(features_prior)
        new_log_likelihood = loglikelihood(y, tree, X, σsq)
        log_α = min(0, new_log_likelihood - old_log_likelihood)
        println("reasign feature")
        println("new log likelihood ", new_log_likelihood)
        println("old log likelihood ", old_log_likelihood)
    elseif action == 2 # reasign operation
        ops_nodes = filter(node -> node.degree > 0, tree)
        isempty(ops_nodes) && return tree # tree containing only a single feature node.
        selected_node = rand(ops_nodes)
        old_tree = deepcopy(selected_node)
        if selected_node.degree == 1
            selected_node.op = rand(1:length(operators[1]))
        elseif selected_node.degree == 2
            selected_node.op = rand(1:length(operators[2]))
        else # lt
            return tree
        end
        new_log_likelihood = loglikelihood(y, tree, X, σsq)
        log_α = min(0, new_log_likelihood - old_log_likelihood)
        println("reasign operation")
        println("new log likelihood ", new_log_likelihood)
        println("old log likelihood ", old_log_likelihood)
    elseif action == 3 # grow tree
        feature_nodes = Node{Float64,3}[]
        depths = Int[]
        get_feature_nodes_with_depth!(feature_nodes, depths, tree, 0)
        selected_index = rand(1:length(feature_nodes))
        selected_node = feature_nodes[selected_index]
        depth = depths[selected_index]
        old_tree = deepcopy(selected_node)
        new_tree = sample_tree_prior(depth, priors)
        while new_tree.degree == 0 # tree containing only a single feature node is not reversible by action 4
            new_tree = sample_tree_prior(depth, priors)
        end
        set_node!(selected_node, new_tree)
        new_log_likelihood = loglikelihood(y, tree, X, σsq)
        new_log_prior_probability = logprobability_tree_prior(tree, 0, priors)
        log_probability_jump_to = -log(length(feature_nodes)) + logprobability_tree_prior(new_tree, depth, priors) - logpdf(depth_prior(depth), true)# minus term to account for trees with single feature node.
        new_ops_nodes = filter(node -> node.degree > 0, tree)
        log_probability_jump_from = -log(length(new_ops_nodes)) + logpdf(features_prior, old_tree.feature)
        log_α = min(0, new_log_likelihood + new_log_prior_probability - old_log_likelihood - old_log_prior_probability - log_probability_jump_to + log_probability_jump_from)
        println("grow tree")
        println("old tree ", old_tree)
        println("new tree ", new_tree)
        println("new log likelihood ", new_log_likelihood)
        println("old log likelihood ", old_log_likelihood)
        println("new_log_prior_probability ", new_log_prior_probability)
        println("old_log_prior_probability ", old_log_prior_probability)
        println("log_probability_jump_to ", log_probability_jump_to)
        println("log_probability_jump_from ", log_probability_jump_from)
    elseif action == 4 # prune tree
        ops_nodes = Node{Float64,3}[]
        depths = Int[]
        get_ops_nodes_with_depth!(ops_nodes, depths, tree, 0)
        isempty(ops_nodes) && return tree # tree containing only a single feature node.
        selected_index = rand(1:length(ops_nodes))
        selected_node = ops_nodes[selected_index]
        depth = depths[selected_index]
        old_tree = deepcopy(selected_node)
        new_tree = Node{Float64,3}(feature=rand(features_prior))
        set_node!(selected_node, new_tree)
        new_log_likelihood = loglikelihood(y, tree, X, σsq)
        new_log_prior_probability = logprobability_tree_prior(tree, 0, priors)
        log_probability_jump_to = -log(length(ops_nodes)) + logpdf(features_prior, new_tree.feature)
        new_feature_nodes = filter(node -> (node.degree == 0 && !node.constant), tree)
        log_probability_jump_from = -log(length(new_feature_nodes)) + logprobability_tree_prior(old_tree, depth, priors) - logpdf(depth_prior(depth), true)# minus term to account for trees with single feature node.
        log_α = min(0, new_log_likelihood + new_log_prior_probability - old_log_likelihood - old_log_prior_probability - log_probability_jump_to + log_probability_jump_from)
        println("prune tree")
        println("old tree ", old_tree)
        println("new tree ", new_tree)
        println("new log likelihood ", new_log_likelihood)
        println("old log likelihood ", old_log_likelihood)
        println("new_log_prior_probability ", new_log_prior_probability)
        println("old_log_prior_probability ", old_log_prior_probability)
        println("log_probability_jump_to ", log_probability_jump_to)
        println("log_probability_jump_from ", log_probability_jump_from)
    elseif action == 5 # collapse operator
        ops_nodes = Node{Float64,3}[]
        depths = Int[]
        get_ops_nodes_with_depth!(ops_nodes, depths, tree, 0)
        isempty(ops_nodes) && return tree # tree containing only a single feature node.
        selected_index = rand(1:length(ops_nodes))
        selected_node = ops_nodes[selected_index]
        depth = depths[selected_index]
        old_tree = deepcopy(selected_node)
        new_tree = if old_tree.degree == 1
            get_child(old_tree, 1)
        elseif old_tree.degree == 2
            children = (get_child(old_tree, 1), get_child(old_tree, 2))
            selected_child = rand((1, 2))
            discarded_child = selected_child == 1 ? children[2] : children[1]
            children[selected_child]
        else # lt
            get_child(old_tree, 1)
        end
        set_node!(selected_node, new_tree)
        new_log_likelihood = loglikelihood(y, tree, X, σsq)
        new_log_prior_probability = logprobability_tree_prior(tree, 0, priors)
        log_probability_jump_to = -log(length(ops_nodes)) - (old_tree.degree == 2 ? log(2) : log(1))
        new_ops_and_feature_nodes = filter(node -> node.degree > 0 || !node.constant, tree)
        correction_multiple_children = if old_tree.degree == 1
            -log(1)
        elseif old_tree.degree == 2
            -log(2) + logprobability_tree_prior(discarded_child, depth + 1, priors) #TODO fill in correct depth
        else
            -log(1) + logpdf(a_prior, get_child(old_tree, 2).val) + logpdf(b_prior, get_child(old_tree, 3).val)
        end
        log_probability_jump_from = -log(length(new_ops_and_feature_nodes)) + logpdf(ops_prior, 1) + correction_multiple_children# TODO generalize 1
        log_α = min(0, new_log_likelihood + new_log_prior_probability - old_log_likelihood - old_log_prior_probability - log_probability_jump_to + log_probability_jump_from)
        println("collapse tree")
        println("new log likelihood ", new_log_likelihood)
        println("old log likelihood ", old_log_likelihood)
        println("new_log_prior_probability ", new_log_prior_probability)
        println("old_log_prior_probability ", old_log_prior_probability)
        println("log_probability_jump_to ", log_probability_jump_to)
        println("log_probability_jump_from ", log_probability_jump_from)
    elseif action == 6 # expand operator
        ops_and_feature_nodes = Node{Float64,3}[]
        depths = Int[]
        get_feature_and_ops_nodes_with_depth!(ops_and_feature_nodes, depths, tree, 0)
        selected_index = rand(1:length(ops_and_feature_nodes))
        selected_node = ops_and_feature_nodes[selected_index]
        depth = depths[selected_index]
        old_tree = deepcopy(selected_node)
        op_index = rand(ops_prior)
        isunary = op_index <= length(operators.ops[1])
        isbinary = length(operators.ops[1]) < op_index <= length(operators.ops[1]) + length(operators.ops[2])
        new_tree = if isunary
            Node{Float64,3}(op=op_index, children=(deepcopy(selected_node),))
        elseif isbinary
            selected_child = rand((1, 2))
            grown_child = sample_tree_prior(depth + 1, priors)
            children = selected_child == 1 ? (deepcopy(selected_node), grown_child) : (grown_child, deepcopy(selected_node))
            return Node{Float64,3}(op=op_index - length(operators.ops[1]), children=children)
        else # assumption lt is only ternary operator
            a = rand(a_prior)
            b = rand(b_prior)
            return Node{Float64,3}(op=1, children=(deepcopy(selected_node), Node{Float64,3}(val=a), Node{Float64,3}(val=b)))
        end
        set_node!(selected_node, new_tree)
        new_log_likelihood = loglikelihood(y, tree, X, σsq)
        new_log_prior_probability = logprobability_tree_prior(tree, 0, priors)
        correction_multiple_children = if isunary
            -log(1)
        elseif isbinary
            -log(2) + logprobability_tree_prior(grown_child, depth + 1, priors)
        else
            -log(1) + logpdf(a_prior, get_child(new_tree, 2).val) + logpdf(b_prior, get_child(new_tree, 3).val)
        end
        log_probability_jump_to = -log(length(ops_and_feature_nodes)) + logpdf(ops_prior, op_index) + correction_multiple_children
        new_ops_nodes = filter(node -> node.degree > 0, tree)
        log_probability_jump_from = -log(length(new_ops_nodes)) - (isbinary ? log(2) : log(1))
        log_α = min(0, new_log_likelihood + new_log_prior_probability - old_log_likelihood - old_log_prior_probability - log_probability_jump_to + log_probability_jump_from)
        println("expand operator")
        println("new log likelihood ", new_log_likelihood)
        println("old log likelihood ", old_log_likelihood)
        println("new_log_prior_probability ", new_log_prior_probability)
        println("old_log_prior_probability ", old_log_prior_probability)
        println("log_probability_jump_to ", log_probability_jump_to)
        println("log_probability_jump_from ", log_probability_jump_from)
    end
    u = rand(Uniform(0, 1))
    accepted = log(u) < log_α
    println("accepted ", accepted, " ", log(u), " ", log_α)
    println(" ")
    if !accepted
        set_node!(selected_node, old_tree)
    end
    return tree
end
my_tree = copy(testo.tree)
discrete(y, X, my_tree, 100.0)
for i in 1:10_000
    println(discrete(y, X, my_tree, 100.0))
end
#= julia> my_tree = copy(testo.tree)
x1 + inv(x2) =#

#= my_tree = Node{Float64,3}(op=1, children=(Node{Float64,3}(feature=2),
    Node{Float64,3}(op=2, children=(Node{Float64,3}(op=2, children=(Node{Float64,3}(feature=2),
        Node{Float64,3}(feature=2))),))))
discrete(y, X, my_tree, priors, 0.1)

while my_tree != Node{Float64,3}(op=2, children=(Node{Float64,3}(feature=2),
    Node{Float64,3}(op=1, children=(Node{Float64,3}(op=2, children=(Node{Float64,3}(feature=1),
        Node{Float64,3}(feature=2))),))))
    discrete(y, X, my_tree, priors, 0.1)
end
#x1 * -(x1 * x2)
 =#
loglikelihood(y, my_tree, X, 100.0)
loglikelihood(y, get_child(my_tree, 1), X, 100.0)

# add two numbers here to test discrete sampling of lt
