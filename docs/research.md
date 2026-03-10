# Research Notes

Date: 2026-03-10

## Scope

UncertainTea aims to be a GPU-native probabilistic programming system in Julia.
Here, "GPU-native" means more than offloading a few linear algebra calls. It means
that the core execution path for `logjoint`, batched chains or particles, and
constraint transforms should be shaped around GPU-friendly execution from the start.

## Executive Summary

- Julia's GPU stack is strong, but it rewards array-oriented and static execution plans
- Existing Julia PPLs prioritize flexibility; that flexibility is expensive to carry into a GPU-native runtime
- Official Gen syntax is more distinctive than the previous draft: `@gen`, tilde syntax, explicit choice addresses, and external conditioning through `choicemap`
- The most plausible direction for UncertainTea is therefore: Gen-like surface syntax, static GPU semantics, and explicit fallback for dynamic behavior

## Official Gen Syntax Notes

After checking the official Gen documentation, the key surface patterns are:

- model definitions use `@gen`
- random choices are written with `~`
- explicit addresses use `{...}` and hierarchical addresses are composed with `=>`
- observations are typically supplied from the outside using `choicemap` and `generate`
- Gen also distinguishes a static modeling language with its own restrictions and combinators

Implication:

- A Gen-like frontend for UncertainTea should not be centered on `@observe`
- A closer match is `@tea (static)` plus tilde syntax and external constraints

Primary references:

- [Gen built-in modeling language](https://www.gen.dev/docs/stable/ref/modeling/)
- [Gen getting started](https://www.gen.dev/docs/stable/tutorials/getting_started/)
- [Gen static modeling language](https://www.gen.dev/docs/stable/ref/modeling/sml/)

## Julia GPU Stack

### CUDA.jl and Metal.jl

- [CUDA.jl](https://cuda.juliagpu.org/stable/) is the main Julia GPU stack for CUDA devices
- [Metal.jl](https://metal.juliagpu.org/stable/) matters for local iteration on Apple Silicon
- Both stacks strongly reward array programming and simple kernels, and punish scalar iteration and irregular runtime structures

Implication:

- A dictionary-backed dynamic trace is a poor fit for the GPU hot path
- Dense layouts such as `theta[param, chain]` or `weight[particle]` should be the default internal representation

### KernelAbstractions.jl

- [KernelAbstractions.jl](https://juliagpu.github.io/KernelAbstractions.jl/stable/) provides a backend-generic kernel layer
- It is most useful for clear parallel kernels such as resampling, reductions, or batched state updates

Implication:

- UncertainTea should start with array operations and only kernelize the actual hotspots

### Enzyme.jl and Reactant.jl

- [Enzyme.jl](https://enzymead.github.io/Enzyme.jl/stable/) provides LLVM-based AD
- [Reactant.jl](https://enzymead.github.io/Reactant.jl/stable/) provides a compiler path through MLIR/XLA-style infrastructure

Implication:

- The safest v1 plan is not "differentiate arbitrary Julia PPL code on the GPU"
- The safer plan is "make `logjoint` easy to differentiate, then add selective AD support where it pays off"

## Existing PPL Ecosystem

### Turing.jl and DynamicPPL

- [Turing.jl](https://turinglang.org/docs/) is the main Julia PPL today
- [DynamicPPL compiler docs](https://turinglang.org/docs/developers/compiler/minituring-compiler/) show a flexible runtime centered around dynamic metadata and model evaluation machinery

Inference:

- Turing's architecture is excellent for flexibility and composition
- It is not the shape I would choose as the core of a GPU-native runtime

### Gen

- [Gen](https://www.gen.dev/docs/stable/) emphasizes generative functions, traces, and programmable inference
- The official docs also separate the static modeling language from the fully dynamic layer

Implication:

- This split is exactly the right mental model for UncertainTea
- We should borrow the syntax and terminology of Gen more directly, while making the static path the main implementation target

### GenJAX and NumPyro

- [GenJAX](https://genjax.gen.dev/) shows how Gen-style ideas can live in a JAX/JIT world
- [NumPyro](https://num.pyro.ai/en/latest/getting_started.html) shows how strongly vectorization and compiled execution matter for high-performance inference

Implication:

- GPU-first PPLs win by static structure, vectorization, and constrained subsets
- They do not win by preserving unrestricted dynamic trace behavior in the hot path

### Stan OpenCL

- [Stan GPU / parallelization docs](https://mc-stan.org/docs/stan-users-guide/parallelization.html) show that practical GPU acceleration often begins with selected batched likelihoods

Implication:

- UncertainTea does not need to solve every inference algorithm on the GPU on day one
- A narrower GPU-first subset is already valuable

## Design Implications for UncertainTea

- Match official Gen syntax more closely: tilde, explicit addresses, external conditioning
- Keep the internal runtime static: precomputed choice layout, dense parameter storage, compiled `logjoint`
- Build a CPU reference path first, then add GPU backends
- Prioritize `ADVI/SVI`, importance sampling, `SIR/SMC`, and then batched `HMC`
- Defer full `NUTS`, unrestricted dynamic traces, and trans-dimensional models

## Source List

- [CUDA.jl](https://cuda.juliagpu.org/stable/)
- [Metal.jl](https://metal.juliagpu.org/stable/)
- [KernelAbstractions.jl](https://juliagpu.github.io/KernelAbstractions.jl/stable/)
- [Enzyme.jl](https://enzymead.github.io/Enzyme.jl/stable/)
- [Reactant.jl](https://enzymead.github.io/Reactant.jl/stable/)
- [Turing.jl](https://turinglang.org/docs/)
- [DynamicPPL compiler documentation](https://turinglang.org/docs/developers/compiler/minituring-compiler/)
- [Gen built-in modeling language](https://www.gen.dev/docs/stable/ref/modeling/)
- [Gen static modeling language](https://www.gen.dev/docs/stable/ref/modeling/sml/)
- [Gen getting started](https://www.gen.dev/docs/stable/tutorials/getting_started/)
- [GenJAX documentation](https://genjax.gen.dev/)
- [NumPyro documentation](https://num.pyro.ai/en/latest/getting_started.html)
- [Stan GPU / parallelization documentation](https://mc-stan.org/docs/stan-users-guide/parallelization.html)
