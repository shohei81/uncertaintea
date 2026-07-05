module UncertainTea

using ForwardDiff
using KernelAbstractions
using LinearAlgebra
using Random
using SpecialFunctions: digamma, loggamma, erf, erfc, erfinv, beta_inc

export @tea
export AddressSpec, ChoiceSpec, ModelSpec
export AddressLiteralPart, AddressDynamicPart
export DistributionSpec, GenerativeCallSpec, RawChoiceRhsSpec
export LoopScopeSpec
export ParameterLayout, ParameterSlotSpec
export ExecutionPlan, ChoicePlanStep
export DeterministicPlanStep, LoopPlanStep
export IdentityTransform, VectorIdentityTransform, LogTransform, LogitTransform, SimplexTransform
export BoundedTransform, LowerBoundedTransform, UpperBoundedTransform
export ChoiceMap, TeaModel, TeaTrace
export StaticMode, DynamicMode
export modelspec, isstaticaddress, isaddresstemplate, isrepeatedchoice, hasrepeatedchoices
export parameterlayout, parametercount, parametervaluecount
export executionplan
export choicemap, generate, assess, logjoint, logjoint_unconstrained, logjoint_gradient_unconstrained
export BackendExecutionPlan, BackendLoweringReport, backend_report, backend_execution_plan
export backend_package_layout, emit_backend_package
export batched_logjoint, batched_logjoint_unconstrained, batched_logjoint_gradient_unconstrained
export BatchedLogjointGradientCache, batched_logjoint_gradient_unconstrained!
export initialparameters, parameter_vector, parameterchoicemap
export transform_to_constrained, transform_to_unconstrained, transform_to_constrained_with_logabsdet
export GPUBackendManifestFile, GPUBackendStageFile, GPUBackendBundleLayout
export GPUBackendCodegenStage, GPUBackendCodegenBundle
export GPUBackendFileEntry, GPUBackendPackageLayout, GPUBackendEmission
export gpu_backend_manifest_file, gpu_backend_stage_files, gpu_backend_bundles, gpu_backend_files
export gpu_backend_codegen_manifest_lines, gpu_backend_codegen_stages, gpu_backend_stage_kind
export gpu_backend_target_name, gpu_backend_module_extension, gpu_backend_module_filename
export gpu_backend_buffer_argument_type, gpu_backend_buffer_argument_declaration
export gpu_backend_argument_signature, gpu_backend_stub_source_lines, gpu_backend_stub_source_blob
export gpu_backend_stage_preamble_lines, gpu_backend_stage_source_lines, gpu_backend_stage_source_blob
export gpu_backend_bundle_layout, gpu_backend_package_layout, gpu_backend_codegen_bundle, gpu_backend_codegen_package_layout
export emit_gpu_backend_package
export HMCChain, HMCChains, HMCMassAdaptationWindowSummary, HMCMassAdaptationSummary, HMCDiagnosticsSummary, HMCParameterSummary, HMCSummary, SamplerWarnings
export ADVIResult, ImportanceSamplingResult, SIRResult, SMCStageSummary, SMCResult
export hmc, hmc_chains, nuts, nuts_chains, batched_hmc, batched_nuts, batched_advi
export batched_importance_sampling, batched_sir, batched_smc
export batched_nuts_package_layout, emit_batched_nuts_package
export tempered_smc_nuts_codegen_bundle, tempered_smc_nuts_package_layout, emit_tempered_smc_nuts_package
export acceptancerate, divergencerate, massadaptationwindows, treedepths, integrationsteps, nchains, numsamples, numstages, rhat, ess, summarize
export check_diagnostics, has_warnings
export posterior_array, parameter_names, to_arviz_dict, to_mcmcchains
export pointwise_loglikelihood, observation_addresses, waic, psis_loo, loo, WAICResult, LOOResult
export map_estimate, laplace_approximation, MAPResult, LaplaceResult
export variational_mean, variational_samples
export predict, PredictiveDraws, addresses, log_evidence
export normal, lognormal, laplace, exponential, gamma, inversegamma, weibull, beta, dirichlet, mvnormal, bernoulli, geometric, negativebinomial, poisson, studentt, categorical
export truncatednormal, truncatedstudentt
export mixture
export device_batched_logjoint, device_batched_logjoint!, device_lowering_report, DeviceBatchedWorkspace, DeviceExecutionPlan
export device_batched_logjoint_gradient, device_batched_logjoint_gradient!
# binomial is intentionally not exported: it would shadow Base.binomial for users.
# Inside @tea models the name resolves to UncertainTea.binomial automatically.

include("ir.jl")
include("core.jl")
include("choicemaps.jl")
include("distributions.jl")
include("runtime.jl")
include("parameters.jl")
include("evaluator.jl")
include("evaluator_pointwise.jl")
include("backend.jl")
include("batched.jl")
include("gpu_backend.jl")
include("inference.jl")
include("frontend.jl")
include("device.jl")

end
