struct Constraint
    features::Array{Symbol, 1}
    fun::Function
end

struct Implication
    condition::Function
    consequence::Function
    condFeatures::Array{Symbol, 1}
    conseqFeatures::Array{Symbol, 1}
end

struct PLAFProgram
    groups::Vector{Tuple{Vararg{Symbol}}}
    constraints::Vector{Constraint}
    implications::Vector{Implication}
end

PLAFProgram() = PLAFProgram(Vector{Tuple}(), Vector{Constraint}(), Vector{Implication}())
initPLAF() = PLAFProgram(Vector{Tuple}(), Vector{Constraint}(), Vector{Implication}())

PLAFProgram(p::PLAFProgram) = PLAFProgram(
    Vector{Tuple}(p.groups),
    Vector{Constraint}(p.constraints),
    Vector{Implication}(p.implications))

Base.empty!(p::PLAFProgram) = begin
    empty!(p.groups)
    empty!(p.constraints)
    empty!(p.implications)
    p
end

function Base.push!(p::PLAFProgram, constraint::Union{Constraint, Implication})
    if constraint isa Constraint
        push!(p.constraints, constraint)
    else
        push!(p.implications, constraint)
    end
    p
end

function Base.show(io::IO, p::PLAFProgram)
    print(io, "PLAFProgram ($(length(p.constraints)) constraints, $(length(p.implications)) implications, $(length(p.groups)) groups)")
end

plaf_helper(x, t) = begin
    constraints = ground.(t)

    quote
        $push!($x, $(constraints...))
    end
end

function group_coverter(t)
    if t isa Expr
        return quote
            Tuple($t)
        end
    end
end

group_helper(x, t) = begin
    if t isa Tuple{Expr}
        groups = group_coverter.(t)
        quote
            $push!($x.groups, $(groups...))
        end
    elseif t isa Tuple{Symbol,Vararg{Symbol}}
        quote
            $push!($x.groups, $t)
        end
    else
        @error "A group should define a list of features not $(typeof(t))"
    end
end

macro PLAF(x, args...)
    esc(plaf_helper(x,args))
end

macro GROUP(x, args...)
    esc(group_helper(x,args))
end

function addkey!(membernames, nam)::Symbol
    if !haskey(membernames, nam)
        membernames[nam] = gensym()
    end
    membernames[nam]
end

onearg(e, f) = e.head == :call && length(e.args) == 2 && e.args[1] == f
mapexpr(f, e) = Expr(e.head, map(f, e.args)...)

replace_syms!(x, membernames; implication=false) = x
replace_syms!(q::QuoteNode, membernames; implication=false) = replace_syms!(Meta.quot(q.value), membernames; implication=implication)

function replace_syms!(e::Expr, membernames; implication=false)
    # println("replace_syms: ", e.head, " ==> ", e)
    if e.head == :quote
        # println("addkey: ", e.args[1], " ==> ", e)
        if !implication
            return addkey!(membernames, Meta.quot(e.args[1]))
        else
            membernames[Meta.quot(e.args[1])] = e.args[1]
            return quote __instance.$(e.args[1]) end
        end
    elseif e.head == :.
        if e.args[1] ∈ (QuoteNode(:cf), QuoteNode(:x_cf), QuoteNode(:counterfactual), :cf, :x_cf, :counterfactual)
            return replace_syms!(e.args[2], membernames; implication=implication)
        elseif e.args[1] ∈ (:x, :inst, :instance, QuoteNode(:x), QuoteNode(:inst), QuoteNode(:instance))
            return quote __orig_instance.$(e.args[2].value) end
        else
            @error "The tuple identifier should be one of (cf, x_cf, counterfactual) or (x, inst, instance), we got $(e.args[1])"
        end
    else
        e2 = mapexpr(x -> replace_syms!(x, membernames; implication=implication), e)
    end
end

function make_source_concrete(x::AbstractVector)
    if isempty(x) || isconcretetype(eltype(x))
        return x
    elseif all(t -> t isa Union{AbstractString, Symbol}, x)
        return Symbol.(x)
    else
        throw(ArgumentError("Column references must be either all the same " *
                            "type or a a combination of `Symbol`s and strings"))
    end
end

function ground(kw)
    grounded_constraints = Pair{Vector{Symbol}, Any}[]

    if kw.head == :if
        # println("Implication: ", kw.head, " -- ", kw.args)

        condfeatures = Dict{Any, Symbol}()
        cond_body::Expr = replace_syms!(kw.args[1], condfeatures; implication=true)
        cond_source::Expr = Expr(:vect, keys(condfeatures)...)
        cond_inputargs::Expr = Expr(:tuple, values(condfeatures)...)

        conseq_features = Dict{Any, Symbol}()
        conseq_body::Expr = replace_syms!(kw.args[2], conseq_features)
        conseq_source::Expr = Expr(:vect, keys(conseq_features)...)
        conseq_inputargs::Expr = Expr(:tuple, values(conseq_features)...)

        conseq_imp_features = Dict{Any, Symbol}()
        conseq_imp_body::Expr = replace_syms!(kw.args[2], conseq_imp_features; implication=true)
        # conseq_imp_source::Expr = Expr(:vect, keys(conseq_imp_features)...)
        # conseq_imp_inputargs::Expr = Expr(:tuple, values(conseq_imp_features)...)

        ## Generate the Implication ...
        generated_func = quote
            $Implication(__orig_instance -> __instance -> ($cond_body && !($conseq_imp_body)),
                __orig_instance -> $conseq_inputargs -> $conseq_body,
                GeCo.make_source_concrete($(cond_source)),
                GeCo.make_source_concrete($(conseq_source)) )
        end

    # elseif kw.head == :generator

    #     println("Generator: ", kw.args[1], " --- ", kw.args[2])
    #     println(kw.head, kw.args)
    #     println(kw.args[1].head, kw.args[1].args)
    #     println(kw.args[2].head, kw.args[2].args)

    #     membernames = Dict{Any, Symbol}()
    #     # gen_body::Expr = replace_syms!(kw.args[2], membernames)
    #     # gen_source::Expr = Expr(:vect, keys(membernames)...)

    #     gen_body = quote all($(kw)) end
    #     gen_source::Expr = Expr(:vect, keys(membernames)...)

    #     println("Gen body: ", gen_body)

    #     generated_func = quote
    #         $Constraint(
    #             GeCo.make_source_concrete($(gen_source)), __orig_instance -> $gen_body)
    #     end

    else
        membernames = Dict{Any, Symbol}()

        body::Expr = replace_syms!(kw, membernames)
        source::Expr = Expr(:vect, keys(membernames)...)
        inputargs::Expr = Expr(:tuple, values(membernames)...)

        generated_func = quote
            $Constraint(GeCo.make_source_concrete($(source)), __orig_instance -> $inputargs -> $body)
        end
    end

    generated_func
end



