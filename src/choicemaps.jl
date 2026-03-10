const Address = Tuple{Vararg{Any}}

mutable struct ChoiceMap
    entries::Vector{Pair{Address,Any}}
end

ChoiceMap() = ChoiceMap(Pair{Address,Any}[])

function choicemap(entries...)
    cm = ChoiceMap()
    if length(entries) == 1 && !_is_choice_entry(entries[1])
        for entry in entries[1]
            _pushchoice!(cm, entry)
        end
        return cm
    end

    for entry in entries
        _pushchoice!(cm, entry)
    end
    return cm
end

_is_choice_entry(entry) = entry isa Pair || (entry isa Tuple && length(entry) == 2)

function normalize_address(address)
    return (address,)
end

normalize_address(address::Symbol) = (address,)
normalize_address(address::Pair) = (normalize_address(address.first)..., normalize_address(address.second)...)
normalize_address(address::QuoteNode) = normalize_address(address.value)

function normalize_address(address::Tuple)
    flattened = ()
    for part in address
        flattened = (flattened..., normalize_address(part)...)
    end
    return flattened
end

function _pushchoice!(cm::ChoiceMap, entry::Pair)
    return _pushchoice!(cm, entry.first, entry.second)
end

function _pushchoice!(cm::ChoiceMap, entry::Tuple)
    length(entry) == 2 || throw(ArgumentError("choice entries must have exactly two elements"))
    return _pushchoice!(cm, entry[1], entry[2])
end

function _pushchoice!(cm::ChoiceMap, address, value)
    normalized = normalize_address(address)
    for idx in eachindex(cm.entries)
        if first(cm.entries[idx]) == normalized
            cm.entries[idx] = normalized => value
            return cm
        end
    end
    push!(cm.entries, normalized => value)
    return cm
end

function Base.getindex(cm::ChoiceMap, address)
    normalized = normalize_address(address)
    for entry in cm.entries
        if first(entry) == normalized
            return last(entry)
        end
    end
    throw(KeyError(normalized))
end

function Base.haskey(cm::ChoiceMap, address)
    normalized = normalize_address(address)
    for entry in cm.entries
        if first(entry) == normalized
            return true
        end
    end
    return false
end

Base.length(cm::ChoiceMap) = length(cm.entries)
Base.isempty(cm::ChoiceMap) = isempty(cm.entries)
Base.iterate(cm::ChoiceMap, state...) = iterate(cm.entries, state...)

function Base.show(io::IO, cm::ChoiceMap)
    print(io, "ChoiceMap(", length(cm.entries), " entries)")
end
