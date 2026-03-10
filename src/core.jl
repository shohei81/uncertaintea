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
