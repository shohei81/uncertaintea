module UncertainTea

using Random

export @tea
export AddressSpec, ChoiceSpec, ModelSpec
export AddressLiteralPart, AddressDynamicPart
export ChoiceMap, TeaModel, TeaTrace
export StaticMode, DynamicMode
export modelspec, isstaticaddress, isaddresstemplate
export choicemap, generate, assess
export normal, bernoulli

include("ir.jl")
include("core.jl")
include("choicemaps.jl")
include("distributions.jl")
include("runtime.jl")
include("frontend.jl")

end
