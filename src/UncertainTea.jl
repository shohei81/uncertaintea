module UncertainTea

using Random

export @tea
export AddressSpec, ChoiceSpec, ModelSpec
export AddressLiteralPart, AddressDynamicPart
export DistributionSpec, GenerativeCallSpec, RawChoiceRhsSpec
export LoopScopeSpec
export ParameterLayout, ParameterSlotSpec
export ExecutionPlan, ChoicePlanStep
export IdentityTransform, LogTransform
export ChoiceMap, TeaModel, TeaTrace
export StaticMode, DynamicMode
export modelspec, isstaticaddress, isaddresstemplate, isrepeatedchoice, hasrepeatedchoices
export parameterlayout, parametercount
export executionplan
export choicemap, generate, assess, logjoint
export initialparameters, parameter_vector, parameterchoicemap
export transform_to_constrained, transform_to_unconstrained
export normal, lognormal, bernoulli

include("ir.jl")
include("core.jl")
include("choicemaps.jl")
include("distributions.jl")
include("runtime.jl")
include("parameters.jl")
include("evaluator.jl")
include("frontend.jl")

end
