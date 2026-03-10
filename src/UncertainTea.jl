module UncertainTea

using Random

export @tea
export AddressSpec, ChoiceSpec, ModelSpec
export AddressLiteralPart, AddressDynamicPart
export DistributionSpec, GenerativeCallSpec, RawChoiceRhsSpec
export LoopScopeSpec
export ParameterLayout, ParameterSlotSpec
export ChoiceMap, TeaModel, TeaTrace
export StaticMode, DynamicMode
export modelspec, isstaticaddress, isaddresstemplate, isrepeatedchoice, hasrepeatedchoices
export parameterlayout, parametercount
export choicemap, generate, assess
export normal, bernoulli

include("ir.jl")
include("core.jl")
include("choicemaps.jl")
include("distributions.jl")
include("runtime.jl")
include("frontend.jl")

end
