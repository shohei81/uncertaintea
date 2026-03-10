abstract type TeaMode end

struct StaticMode <: TeaMode end
struct DynamicMode <: TeaMode end

const STATIC_MODE = StaticMode()
const DYNAMIC_MODE = DynamicMode()

struct TeaModel{M<:TeaMode,F,S}
    mode::M
    name::Symbol
    impl::F
    spec::S
    evaluator_cache::Base.RefValue{Any}
    backend_cache::Base.RefValue{Any}
end

function TeaModel(mode::M, name::Symbol, impl::F, spec::S) where {M<:TeaMode,F,S}
    return TeaModel{M,F,S}(mode, name, impl, spec, Ref{Any}(nothing), Ref{Any}(nothing))
end

struct TeaCall{M<:TeaModel,A<:Tuple}
    model::M
    args::A
end

(model::TeaModel)(args...) = TeaCall(model, args)

function Base.show(io::IO, model::TeaModel)
    print(io, "TeaModel(", model.name, ", mode=", nameof(typeof(model.mode)))
    model.spec === nothing || print(io, ", choices=", length(model.spec.choices))
    print(io, ")")
end
