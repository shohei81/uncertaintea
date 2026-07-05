module UncertainTeaMCMCChainsExt

using UncertainTea
using MCMCChains

# Add the ::HMCChains method to the core `to_mcmcchains` function (which is
# declared method-less in UncertainTea). Uses only the stable
# `Chains(array, names)` constructor.
function UncertainTea.to_mcmcchains(chains::UncertainTea.HMCChains; space::Symbol=:constrained)
    array = UncertainTea.posterior_array(chains; space=space)
    names = Symbol.(UncertainTea.parameter_names(chains; space=space))
    return MCMCChains.Chains(array, names)
end

end # module
