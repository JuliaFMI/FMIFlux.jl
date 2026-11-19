#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

import FMIImport.FMIBase:
    assert_integrator_valid,
    isdual,
    istracked,
    issense,
    undual,
    unsense,
    unsense_copy,
    untrack,
    FMUSnapshot,
    getStates,
    getContinuousStates,
    getEmptyReal,
    handleEvents
import FMIImport: finishSolveFMU, prepareSolveFMU, snapshot_or_update!
import Optim
import FMIImport.FMIBase.ProgressMeter
import FMISensitivity.SciMLSensitivity.SciMLBase:
    CallbackSet,
    ContinuousCallback,
    ODESolution,
    ReturnCode,
    RightRootFind,
    VectorContinuousCallback,
    set_u!,
    terminate!,
    u_modified!,
    build_solution
using FMISensitivity.ReverseDiff: TrackedArray
import FMISensitivity.SciMLSensitivity:
    InterpolatingAdjoint, ReverseDiffVJP, AutoForwardDiff
import ThreadPools
import FMIImport.FMIBase

using FMIImport.FMIBase.DiffEqCallbacks
using FMIImport.FMIBase.SciMLBase: ODEFunction, ODEProblem, solve
using FMIImport.FMIBase:
    fmi2ComponentState,
    fmi2ComponentStateContinuousTimeMode,
    fmi2ComponentStateError,
    fmi2ComponentStateEventMode,
    fmi2ComponentStateFatal,
    fmi2ComponentStateInitializationMode,
    fmi2ComponentStateInstantiated,
    fmi2ComponentStateTerminated,
    fmi2StatusOK,
    fmi2Type,
    fmi2TypeCoSimulation,
    fmi2TypeModelExchange,
    logError,
    logInfo,
    logWarning,
    fast_copy!,
    isTrue,
    isContinuousTimeMode,
    ERR_MSG_CONT_TIME_MODE,
    setupSolver!
using FMISensitivity.SciMLSensitivity:
    ForwardDiffSensitivity, InterpolatingAdjoint, ReverseDiffVJP, ZygoteVJP
import DifferentiableEigen
import DifferentiableEigen.LinearAlgebra: I, norm, pinv

import FMIImport.FMIBase: EMPTY_fmi2Real, EMPTY_fmi2ValueReference
import FMIImport.FMIBase
import FMISensitivity: NoTangent, ZeroTangent

DEFAULT_PROGRESS_DESCR = "Simulating ME-NeuralFMU ..."
DEFAULT_CHUNK_SIZE = 32
DUMMY_DT = 1/10
EVENT_SENSITIVITY_STATE_ATOL = 1e-9
EVENT_SENSITIVITY_STATE_RTOL = 1e-8
EVENT_SENSITIVITY_TIME_ATOL = 1e-9
EVENT_SALTATION_DENOM_ATOL = 1e-10
EVENT_SENSITIVITY_JACOBIAN_MAX = 1e6
# ANN in front of the FMU (`modifiedState`): finite-difference step and relative tolerance
# for mapping the FMU state after an event back to the ODE state
MODIFIED_STATE_FD_STEP = 1e-6
MODIFIED_STATE_RTOL = 1e-14 # event indicators can sit ~1e-10 next to zero after an event
MODIFIED_STATE_WARN_RTOL = 1e-6 # warn if the found state is further away than this
FORCE_SCALAR_EVENT_CALLBACKS_KEY = :FMIFlux_force_scalar_event_callbacks

function forceScalarEventCallbacks()
    try
        return task_local_storage(FORCE_SCALAR_EVENT_CALLBACKS_KEY) === true
    catch err
        err isa KeyError || rethrow()
        return false
    end
end

function withScalarEventCallbacks(f)
    return task_local_storage(f, FORCE_SCALAR_EVENT_CALLBACKS_KEY, true)
end

"""
The mutable struct representing an abstract (simulation mode unknown) NeuralFMU.
"""
abstract type NeuralFMU end

"""
Structure definition for a NeuralFMU, that runs in mode `Model Exchange` (ME).
"""
mutable struct ME_NeuralFMU{M,R} <: NeuralFMU

    model::M
    p::AbstractArray{<:Real}
    re::R
    solvekwargs::Any

    re_model::Any
    re_p::Any

    fmu::FMU

    tspan::Any
    saveat::Any
    saved_values::Any
    recordValues::Any
    solver::Any

    valueStack::Any

    customCallbacksBefore::Array
    customCallbacksAfter::Array

    x0::Union{Array{Float64},Nothing}
    firstRun::Bool

    tolerance::Union{Real,Nothing}
    parameters::Union{Dict{<:Any,<:Any},Nothing}
    inputs::Union{Dict{<:Any,<:Any},Nothing}

    modifiedState::Bool

    execution_start::Real

    condition_buffer::Union{AbstractArray{<:Real},Nothing}

    snapshots::Bool

    function ME_NeuralFMU{M,R}(model::M, p::AbstractArray{<:Real}, re::R) where {M,R}
        inst = new()
        inst.model = model
        inst.p = p
        inst.re = re
        inst.x0 = nothing
        inst.saveat = nothing

        inst.re_model = nothing
        inst.re_p = nothing

        inst.inputs = nothing

        inst.modifiedState = false

        # inst.startState = nothing 
        # inst.stopState = nothing
        # inst.startEventInfo = nothing 
        # inst.stopEventInfo = nothing

        inst.customCallbacksBefore = []
        inst.customCallbacksAfter = []

        inst.execution_start = 0.0

        inst.condition_buffer = nothing

        inst.snapshots = false

        return inst
    end
end

"""
Structure definition for a NeuralFMU, that runs in mode `Co-Simulation` (CS).
"""
mutable struct CS_NeuralFMU{F,C} <: NeuralFMU
    model::Any
    fmu::F

    tspan::Any

    p::Union{AbstractArray{<:Real},Nothing}
    re::Any # restructure function

    snapshots::Bool

    function CS_NeuralFMU{F,C}() where {F,C}
        inst = new{F,C}()

        inst.re = nothing
        inst.p = nothing

        inst.snapshots = false

        return inst
    end
end

function dummyDynamics(x_d, x, dx, t)
    x_d_round = round(Integer, x_d)

    ẋ_d = 0.0

    if DUMMY_DT > 0.0
        ẋ_d = 1/3 * (2 * π * 1/DUMMY_DT) * cos(t * 2 * π * 1/DUMMY_DT) # - 2 * (x_d - x_d_round) * 1e-3 # sin(x[1] + t) * 1e-2 + cos(dx[1]) * 1e-2 
    end

    return ẋ_d
end

function evaluateModel(
    nfmu::ME_NeuralFMU,
    c::FMU2Component,
    x;
    p = nfmu.p,
    t = c.default_t,
    force::Bool = false,
)
    @assert getCurrentInstance(nfmu.fmu) == c "Thread `$(Threads.threadid())` wants to evaluate wrong component!"

    # [ToDo]: Skip array check, e.g. by using a flag
    #if p !== nfmu.re_p || p != nfmu.re_p # || isnothing(nfmu.re_model)
    #    nfmu.re_p = p # fast_copy!(nfmu, :re_p, p)
    #    nfmu.re_model = nfmu.re(p)
    #end
    #return nfmu.re_model(x)

    #@debug "evaluateModel(t=$(t)) [out-of-place dx]"

    x_d = nothing

    if c.fmu.isDummyDiscrete
        # TODO: Non-sensitive augmented states don't work with reversediff ... leads to NaNs ... this was really hard to debug BTW ...
        # Because we only want to hard-set it for now (pick FMUState)
        c.default_x_d = unsense(x[end:end])
        x_d = x[end]
        x = x[1:(end-1)]
    end

    mode = c.force
    c.force = force

    #nfmu.p = p 
    c.default_t = t
    dx = FMIFlux.evaluate(nfmu, x; p = p)

    c.force = mode

    if c.fmu.isDummyDiscrete
        c.default_x_d = getEmptyReal(c)
        nx = length(nfmu.fmu.modelDescription.stateValueReferences)
        #dx_y = vcat(dx_y[1:nx], 1e-2 * cos(sum(x)), dx_y[nx+1:end])
        ẋ_d = dummyDynamics(x_d, x[1:nx], dx[1:nx], t)
        if dx isa FMIBase.FMUADOutput
            # the FMU output is returned directly (no ANN after the FMU), but
            # `vcat(::FMUADOutput, ::ReverseDiff.TrackedReal)` is ambiguous
            dx = [dx..., ẋ_d]
        else
            dx = vcat(dx, ẋ_d)
        end
    end

    return dx
end

function evaluateModel(
    nfmu::ME_NeuralFMU,
    c::FMU2Component,
    dx,
    x;
    p = nfmu.p,
    t = c.default_t,
    force::Bool = false,
)
    @assert getCurrentInstance(nfmu.fmu) == c "Thread `$(Threads.threadid())` wants to evaluate wrong component!"

    # [ToDo]: Skip array check, e.g. by using a flag
    #if p !== nfmu.re_p || p != nfmu.re_p # || isnothing(nfmu.re_model)
    #    nfmu.re_p = p # fast_copy!(nfmu, :re_p, p)
    #    nfmu.re_model = nfmu.re(p)
    #end
    #dx[:] = nfmu.re_model(x)

    #@debug "evaluateModel(t=$(t)) [in-place dx]"

    #nfmu.p = p 

    x_d = nothing
    if c.fmu.isDummyDiscrete
        # ToDo: because we only want to hard-set it for now (pick FMUState)
        c.default_x_d = unsense(x[end:end])
        x_d = x[end]
        x = x[1:(end-1)]
    end

    mode = c.force
    c.force = force

    c.default_t = t

    # Info: This may be for implicit solvers!
    #@info "dx: $(typeof(dx))"
    #@info "x: $(typeof(x))"

    #if issense(x) && !issense(dx)
    #    x = unsense(x)
    #end

    if c.fmu.isDummyDiscrete
        c.default_x_d = getEmptyReal(c)
        nx = length(nfmu.fmu.modelDescription.stateValueReferences)
        #tmp = FMIFlux.evaluate(nfmu, x; p=p)
        #dx_y[1:nx] = tmp[1:nx]
        #dx_y[nx+1] = 1e-2 * cos(sum(x)) 
        #dx_y[nx+2:end] = tmp[nx+1:end]
        dx[1:nx] = FMIFlux.evaluate(nfmu, x; p = p)
        dx[nx+1] = dummyDynamics(x_d, x[1:nx], dx[1:nx], t)
    else
        dx[:] = FMIFlux.evaluate(nfmu, x; p = p)
    end

    c.force = mode

    return nothing
end

# function evaluateModel(nfmu::ME_NeuralFMU, c::FMU2Component, x; p = nfmu.p, t = c.default_t, force::Bool=false)
#     @assert getCurrentInstance(nfmu.fmu) == c "Thread `$(Threads.threadid())` wants to evaluate wrong component!"

#     nfmu.p = p 

#     @assert false "oop"
# end

# function evaluateModel(
#     nfmu::ME_NeuralFMU,
#     c::FMU2Component,
#     dx,
#     x;
#     p = nfmu.p,
#     t = c.default_t, force::Bool=false)

#     nfmu.p = p 

#     if nfmu.fmu.isDummyDiscrete
#         dx[1:end-1] = FMIFlux.evaluate(nfmu, x[1:end-1]; p=p)
#         dx[end] = 1e-2 * cos(sum(x)) # TODO: Non-sensitive augmented states don't work with reversediff ... leads to NaNs ... this was really hard to debug BTW ...
#     else
#         dx[:] = FMIFlux.evaluate(nfmu, x; p=p)
#     end

#     return nothing
# end

##### EVENT HANDLING START

# Parameters for evaluating the ANN inside solver callbacks (event conditions/affects,
# step completion, saving). Callbacks must use the parameters the solver is currently
# integrating with (`integrator.p`), not `nfmu.p`: during a sensitivity solve such as
# `ReverseDiffAdjoint` the solver re-tracks the parameters on its own tape, while
# `nfmu.p` still belongs to the caller's (outer) tape. Recording onto that outer tape
# mixes tapes and, with ReverseDiff >= 1.18, throws once the outer tape has finished
# recording (e.g. when `ReverseDiff.jacobian` replays its forward pass).
# Falls back to `nfmu.p` if no full integrator is available, e.g. when callbacks are
# called directly (tests pass `nothing` or a `NamedTuple`).
function callbackParameters(nfmu::ME_NeuralFMU, integrator)
    if !isnothing(integrator) && hasproperty(integrator, :p)
        return integrator.p
    end
    return nfmu.p
end

function invalidateStaleDummyDiscreteSnapshots!(
    nfmu::ME_NeuralFMU,
    c::FMU2Component,
    writeSnapshot,
    readSnapshot,
)
    if !nfmu.fmu.isDummyDiscrete || !istracked(nfmu.p)
        return nothing
    end

    if !isnothing(writeSnapshot) || !isnothing(readSnapshot)
        return nothing
    end

    # ReverseDiff uses the dummy discrete state as an extra ODE coordinate, so
    # FMIBase.eval_set! later restores FMU states by searching c.snapshots for
    # the current x_d. Those dummy values are local to a single solve (0, 1, 2,
    # ...), therefore a stale snapshot from an older optimization step can have
    # the same x_d while representing different neural-network parameters. Keep
    # explicit read/write snapshots intact above, but invalidate the component
    # cache before a fresh adjoint solve so only snapshots from this solve can be
    # selected during the ReverseDiff replay.
    freed = 0
    for snapshot in c.snapshots
        if snapshot.valid
            FMIBase.freeSnapshot!(snapshot)
            freed += 1
        end
    end

    if freed > 0
        @debug "Invalidated $(freed) stale dummy-discrete snapshots before ReverseDiff solve."
    end

    return nothing
end

function startCallback(
    integrator,
    nfmu::ME_NeuralFMU,
    c::Union{FMU2Component,Nothing},
    t_start,
    writeSnapshot,
    readSnapshot,
)

    ignore_derivatives() do

        nfmu.execution_start = time()

        t_start = unsense(t_start)

        @assert t_start == nfmu.tspan[1] "startCallback(...): Called for non-start-point t=$(t)"

        x0 = nfmu.x0

        if nfmu.fmu.isDummyDiscrete
            x0 = x0[1:(end-1)]
        end

        c, x0 = prepareSolveFMU(
            nfmu.fmu,
            c,
            fmi2TypeModelExchange;
            parameters = nfmu.parameters,
            t_start = t_start,
            t_stop = nfmu.tspan[end],
            tolerance = nfmu.tolerance,
            x0 = x0,
            inputs = nfmu.inputs,
            #handleEvents = FMIFlux.handleEvents,
            cleanup = true,
        )

        if nfmu.fmu.isDummyDiscrete
            c.x_d = nfmu.x0[end:end]
        end

        invalidateStaleDummyDiscreteSnapshots!(nfmu, c, writeSnapshot, readSnapshot)

        if c.eventInfo.nextEventTime == t_start &&
           c.eventInfo.nextEventTimeDefined == fmi2True
            @debug "Initial time event detected!"
        else
            @debug "No initial time events ..."
        end

        # if nfmu.snapshots
        #     sn = FMIBase.snapshot!(c.solution, 0)
        #     @assert sn.t == t_start "Initial snapshot is not at initial time $(sn.t) != $(t_start)"
        # end

        if !isnothing(writeSnapshot)

            if t_start != writeSnapshot.t
                logWarning(
                    c.fmu,
                    "Snapshot time mismatch for `writeSnapshot`, snapshot time = $(writeSnapshot.t), but start time is $(t_start)",
                )
            end

            FMIBase.update!(c, writeSnapshot) # ; t = t_start)

            @assert writeSnapshot.t == t_start "After writing the snapshot, snapshot time $(writeSnapshot.t) doesn't match start time $(t_start)."
        end

        if !isnothing(readSnapshot)
            @assert c == readSnapshot.instance "Snapshot instance mismatch, snapshot instance is $(readSnapshot.instance.addr), current component is $(c.addr)"
            # c = readSnapshot.instance 

            if t_start != readSnapshot.t
                logWarning(
                    c.fmu,
                    "Snapshot time mismatch for `readSnapshot`, snapshot time = $(readSnapshot.t), but start time is $(t_start)",
                )
            end

            @debug "ME_NeuralFMU: Applying snapshot..."

            # this is a very special case:
            # sometimes, snasphsots are recorded with dummyDiscrete=true,
            # but later used with dummydiscrete=false (e.g. LossAccumulationScheduler).
            # Therefore, we manually co
            FMIBase.apply!(c, readSnapshot; t = t_start)
            @debug "ME_NeuralFMU: Snapshot applied."
        end

    end

    #@warn "Experimental -> c.force = true"
    #c.force = true

    return c
end

function stopCallback(nfmu::ME_NeuralFMU, c::FMU2Component, t)

    @assert getCurrentInstance(nfmu.fmu) == c "Thread `$(Threads.threadid())` wants to evaluate wrong component!"

    ignore_derivatives() do

        t = unsense(t)

        @assert t == nfmu.tspan[end] "stopCallback(...): Called for non-start-point t=$(t)"
    end

    return c
end

# Read next time event from fmu and provide it to the integrator 
function time_choice(nfmu::ME_NeuralFMU, c::FMU2Component, integrator, tStart, tStop)

    @assert isContinuousTimeMode(c) "time_choice(...):\n" * ERR_MSG_CONT_TIME_MODE
    @assert getCurrentInstance(nfmu.fmu) == c "Thread `$(Threads.threadid())` wants to evaluate wrong component!"
    @assert c.fmu.executionConfig.handleTimeEvents "time_choice(...) was called, but execution config disables time events.\nPlease open a issue."
    # assert_integrator_valid(integrator)

    # last call may be after simulation end
    if c == nothing
        return nothing
    end

    c.solution.evals_timechoice += 1

    if isTrue(c.eventInfo.nextEventTimeDefined)

        if c.eventInfo.nextEventTime >= tStart && c.eventInfo.nextEventTime <= tStop
            @debug "time_choice(...): At $(integrator.t) next time event announced @$(c.eventInfo.nextEventTime)s"
            return c.eventInfo.nextEventTime
        else
            # the time event is outside the simulation range!
            @debug "Next time event @$(c.eventInfo.nextEventTime)s is outside simulation time range ($(tStart), $(tStop)), skipping."
            return nothing
        end
    else
        return nothing
    end
end

# [ToDo] for now, ReverseDiff (together with the rrule) seems to have a problem with the SubArray here (when `collect` it accesses array elements that are #undef), 
#        so I added an additional (single allocating) dispatch...
#        Type is ReverseDiff.TrackedReal{Float64, Float64, ReverseDiff.TrackedArray{Float64, Float64, 1, Vector{Float64}, Vector{Float64}}}[#undef, #undef, #undef, ...]
function condition!(
    nfmu::ME_NeuralFMU,
    c::FMU2Component,
    out::AbstractArray{<:ReverseDiff.TrackedReal},
    x,
    t,
    integrator,
    handleEventIndicators,
)

    if !isassigned(out, 1)
        if isnothing(nfmu.condition_buffer)
            logInfo(
                nfmu.fmu,
                "There is currently an issue with the condition buffer pre-allocation, the buffer can't be overwritten by the generated rrule.\nBuffer is generated automatically.",
            )
            @assert length(out) == length(handleEventIndicators) "Number of event indicators to handle ($(handleEventIndicators)) doesn't fit buffer size $(length(out))."
            nfmu.condition_buffer = zeros(eltype(out), length(out))
        elseif eltype(out) != eltype(nfmu.condition_buffer) ||
               length(out) != length(nfmu.condition_buffer)
            nfmu.condition_buffer = zeros(eltype(out), length(out))
        end
        out[:] = nfmu.condition_buffer
    end

    invoke(
        condition!,
        Tuple{ME_NeuralFMU,FMU2Component,Any,Any,Any,Any,Any},
        nfmu,
        c,
        out,
        x,
        t,
        integrator,
        handleEventIndicators,
    )

    return nothing
end

function condition!(
    nfmu::ME_NeuralFMU,
    c::FMU2Component,
    out,
    x,
    t,
    integrator,
    handleEventIndicators,
)

    @assert isContinuousTimeMode(c) "condition!(...):\n" * ERR_MSG_CONT_TIME_MODE

    @assert getCurrentInstance(nfmu.fmu) == c "Thread `$(Threads.threadid())` wants to evaluate wrong component!"
    @assert c.state == fmi2ComponentStateContinuousTimeMode "condition!(...):\n" *
                                                            FMIBase.ERR_MSG_CONT_TIME_MODE

    # [ToDo] Evaluate on light-weight model (sub-model) without fmi2GetXXX or similar and the bottom ANN.
    #        Basically only the layers from very top to FMU need to be evaluated here.

    prev_t = c.default_t
    prev_ec = c.default_ec
    prev_ec_idcs = c.default_ec_idcs

    c.default_t = t
    c.default_ec = out
    c.default_ec_idcs = handleEventIndicators

    #@info "$(c.default_ec)"

    #@info "x: $(unsense(x))" * (issense(x) ? " (sens.)" : "")

    evaluateModel(nfmu, c, x; p = callbackParameters(nfmu, integrator), t = t)

    #@info "out: $(unsense(out))" * (issense(out) ? " (sens.)" : "")

    # write back to condition buffer
    if (!isdual(out) && isdual(c.output.ec)) || (!istracked(out) && istracked(c.output.ec))
        @warn "writing sensitive condition to unsensitive buffer."
        out[:] = unsense(c.output.ec)
    elseif (isdual(out) && !isdual(c.output.ec)) ||
           (istracked(out) && !istracked(c.output.ec))
        @assert false "writing unsensitive condition to sensitive buffer."
    else
        out[:] = c.output.ec # [ToDo] This seems not to be necessary, because of `c.default_ec = out`
    end

    # reset
    c.default_t = prev_t
    c.default_ec = prev_ec
    c.default_ec_idcs = prev_ec_idcs

    c.solution.evals_condition += 1

    # #if issense(x) || issense(t)
    #     JVP = [1 0 0; 1 0 0] * x #c.∂e_∂x.mtx * x
    #     C = unsense(c.output.ec) - unsense(JVP)
    #     out[:] = JVP + C
    # #else
    # #    out[:] = c.output.ec
    # #end

    @debug "condition!(...) -> [typeof=$(typeof(out))]\n$(unsense(out))"

    return nothing
end

global lastIndicator = nothing
global lastIndicatorX = nothing
global lastIndicatorT = nothing
function conditionSingle(nfmu::ME_NeuralFMU, c::FMU2Component, index, x, t, integrator)

    @assert isContinuousTimeMode(c) "condition!(...):\n" * ERR_MSG_CONT_TIME_MODE

    @assert getCurrentInstance(nfmu.fmu) == c "Thread `$(Threads.threadid())` wants to evaluate wrong component!"

    if c.fmu.handleEventIndicators != nothing && index ∉ c.fmu.handleEventIndicators
        return 1.0
    end

    global lastIndicator

    if lastIndicator == nothing ||
       length(lastIndicator) != c.fmu.modelDescription.numberOfEventIndicators
        lastIndicator = zeros(c.fmu.modelDescription.numberOfEventIndicators)
    end

    prev_t = c.default_t
    prev_ec = c.default_ec
    prev_ec_idcs = c.default_ec_idcs

    # [ToDo] Evaluate on light-weight model (sub-model) without fmi2GetXXX or similar and the bottom ANN
    try
        c.default_t = t
        c.default_ec = lastIndicator
        c.default_ec_idcs =
            collect(UInt32(i) for i = 1:c.fmu.modelDescription.numberOfEventIndicators)

        evaluateModel(nfmu, c, x; p = callbackParameters(nfmu, integrator), t = t)
    finally
        c.default_t = prev_t
        c.default_ec = prev_ec
        c.default_ec_idcs = prev_ec_idcs
    end

    c.solution.evals_condition += 1

    return lastIndicator[index]
end

function continuousStateLength(c)
    return length(c.fmu.modelDescription.stateValueReferences)
end

function continuousStateView(c, x)
    nx = continuousStateLength(c)
    return view(x, 1:nx)
end

function continuousEventIsNoop(c, left_x, right_x)
    return isapprox(
        continuousStateView(c, left_x),
        continuousStateView(c, right_x);
        atol = EVENT_SENSITIVITY_STATE_ATOL,
        rtol = EVENT_SENSITIVITY_STATE_RTOL,
    )
end

function isNearCoincidentNoopEvent(c, t, left_x, right_x)
    if isempty(c.solution.events) || !continuousEventIsNoop(c, left_x, right_x)
        return false
    end

    previous = c.solution.events[end]
    return abs(t - previous.t) <= EVENT_SENSITIVITY_TIME_ATOL
end

function eventTransitionJacobianIsUsable(jac)
    if !all(isfinite, jac)
        return false
    end

    # The event reset map for an FMI state event should be locally bounded. A
    # sampled Jacobian above this limit is almost always a root-side selection
    # artifact (sampling on the wrong side of a discontinuity), and feeding it
    # into ReverseDiffAdjoint creates the observed 1e10-scale gradients.
    return maximum(abs.(jac)) <= EVENT_SENSITIVITY_JACOBIAN_MAX
end

function smoothmax(vec::AbstractVector; alpha = 0.5)
    dividend = 0.0
    divisor = 0.0
    e = Float64(ℯ)
    for x in vec
        dividend += x * e^(alpha * x)
        divisor += e^(alpha * x)
    end
    return dividend / divisor
end

function smoothmax(a, b; kwargs...)
    return smoothmax([a, b]; kwargs...)
end

# With an ANN in front of the FMU (`nfmu.modifiedState`), the (continuous) ODE state `x` is
# mapped onto the FMU state `z = g(x)` before the FMU is evaluated. The following helpers
# evaluate `g` and its Jacobian.
#
# `g` doesn't depend on a dummy discrete state, so the NeuralFMU is evaluated on the
# continuous states only: evaluating with a dummy discrete state applies the snapshot that
# belongs to it, which would reset the FMU to the state before the event while handling it.
function evaluateModelContinuous(nfmu::ME_NeuralFMU, c::FMU2Component, x_c; kwargs...)
    dummyDiscrete = c.fmu.isDummyDiscrete
    c.fmu.isDummyDiscrete = false
    try
        return evaluateModel(nfmu, c, x_c; kwargs...)
    finally
        c.fmu.isDummyDiscrete = dummyDiscrete
    end
end

# `g(x)`, the FMU state is read back from the FMU, so this works independent of the outputs
# the FMU call inside the NeuralFMU requests.
function modifiedStateMap(nfmu::ME_NeuralFMU, c::FMU2Component, x_c, t)
    evaluateModelContinuous(nfmu, c, x_c; p = unsense(nfmu.p), t = t, force = true)
    return getContinuousStates(c)
end

# `∂g/∂x` by central finite differences, `g` is the (smooth) ANN part in front of the FMU
function modifiedStateJacobian(nfmu::ME_NeuralFMU, c::FMU2Component, x_c, t)
    nz = continuousStateLength(c)
    G = zeros(nz, length(x_c))
    for i = 1:length(x_c)
        h = MODIFIED_STATE_FD_STEP * max(1.0, abs(x_c[i]))
        xp = copy(x_c)
        xm = copy(x_c)
        xp[i] += h
        xm[i] -= h
        G[:, i] =
            (modifiedStateMap(nfmu, c, xp, t) .- modifiedStateMap(nfmu, c, xm, t)) ./ (2h)
    end
    return G
end

# Searches the ODE state `x` with `g(x) = z_target` (Gauss-Newton with backtracking, least
# squares if `g` can't reach `z_target` exactly), starting at `x_guess`. Leaves the FMU at
# the state belonging to the returned `x` and returns `x` together with the residual norm.
function solveModifiedState(
    nfmu::ME_NeuralFMU,
    c::FMU2Component,
    z_target,
    x_guess,
    t;
    maxiters::Integer = 50,
)
    x = collect(Float64, x_guess)
    r = modifiedStateMap(nfmu, c, x, t) .- z_target
    res = norm(r)
    tol = MODIFIED_STATE_RTOL * (1.0 + norm(z_target))

    for _ = 1:maxiters
        res <= tol && break

        G = modifiedStateJacobian(nfmu, c, x, t)
        Δx = -(pinv(G) * r)

        improved = false
        α = 1.0
        while α >= 1e-6
            x_new = x .+ α .* Δx
            r_new = modifiedStateMap(nfmu, c, x_new, t) .- z_target
            res_new = norm(r_new)
            if res_new < res
                x, r, res = x_new, r_new, res_new
                improved = true
                break
            end
            α /= 2.0
        end
        improved || break
    end

    # the last evaluation might have been a rejected step, so set the FMU to `x`
    modifiedStateMap(nfmu, c, x, t)

    return x, res
end

# `g(x; p)` with sensitivities (AD types in `x` and/or `p`): the FMU returns its continuous
# states as output `y` (via the instance default `y_refs`), which is hidden from the ANN
# part behind the FMU.
function modifiedStateOutput(nfmu::ME_NeuralFMU, c::FMU2Component, x_c, p, t)
    stateRefs = c.fmu.modelDescription.stateValueReferences

    prev_y_refs = c.default_y_refs
    prev_y = c.default_y
    prev_show_y = c.output.show_y

    c.default_y_refs = stateRefs
    c.default_y = zeros(fmi2Real, length(stateRefs))
    c.output.show_y = false

    z = nothing
    try
        evaluateModelContinuous(nfmu, c, x_c; p = p, t = t, force = true)
        @assert c.output.len_y == length(stateRefs) "The FMU did not return its states as outputs. Event sensitivities with an ANN in front of the FMU require an FMU call without `y_refs` inside the NeuralFMU (and FMIBase >= 1.6.4)."
        z = collect(c.output.y)
    finally
        c.default_y_refs = prev_y_refs
        c.default_y = prev_y
        c.output.show_y = prev_show_y
    end

    return z
end

# With an ANN in front of the FMU, the state after the event `x_r` solves
# `g_p(x_r) = R(g_p(x_l))`, so it depends on the state before the event `x_l` and directly
# on the ANN parameters `p`:
#   dx_r = pinv(∂g/∂x(x_r)) * (∂R * (∂g/∂x(x_l) * dx_l + ∂g/∂p(x_l) * dp) - ∂g/∂p(x_r) * dp)
# `jac_fmu` is the reset Jacobian `∂R` sampled in FMU coordinates, the Jacobians of `g` are
# determined on the primal values (ForwardDiff). Like for the event Jacobian without an ANN
# in front of the FMU, the result is applied as linearization to the sensitive integrator
# state and parameters, so no FMU evaluation is recorded on an AD tape inside the callback.
# Returns the new (sensitive) integrator state, `right_x` is its value.
function modifiedStateEventSensitivity(
    nfmu::ME_NeuralFMU,
    c::FMU2Component,
    integrator,
    right_x,
    jac_fmu,
    t,
)
    nx = continuousStateLength(c)

    u = integrator.u
    p = callbackParameters(nfmu, integrator)

    left_x_c = unsense(u[1:nx])
    right_x_c = right_x[1:nx]
    p0 = unsense(p)

    A = pinv(modifiedStateJacobian(nfmu, c, right_x_c, t)) # pinv(∂g/∂x(x_r))
    R = jac_fmu[1:nx, 1:nx]

    G_left = ForwardDiff.jacobian(_x -> modifiedStateOutput(nfmu, c, _x, p0, t), left_x_c)
    dx = (A * R * G_left) * (u[1:nx] .- left_x_c)

    if issense(p)
        P_left =
            ForwardDiff.jacobian(_p -> modifiedStateOutput(nfmu, c, left_x_c, _p, t), p0)
        P_right =
            ForwardDiff.jacobian(_p -> modifiedStateOutput(nfmu, c, right_x_c, _p, t), p0)
        dx = dx .+ (A * (R * P_left .- P_right)) * (p .- p0)
    end

    new_u = right_x_c .+ dx
    if c.fmu.isDummyDiscrete
        # the dummy discrete state is passed through (unit sensitivity)
        new_u = vcat(new_u, right_x[end] + (u[end] - unsense(u[end])))
    end

    # set the FMU back to the state belonging to `right_x`
    modifiedStateMap(nfmu, c, right_x[1:nx], t)

    return new_u
end

# Sample the reset map x_right = R(x_left) around an already-handled FMU event.
# The returned matrix maps perturbations of the left state to perturbations of the
# right state. With a dummy discrete state, the final row/column keeps the dummy
# coordinate differentiable while continuous-state entries are sampled from the FMU.
function sampleStateChangeJacobian(nfmu, c, event::FMUEvent; step = 1e-8)

    left_x_c = event.left_snapshot.x_c
    right_x_c = event.right_snapshot.x_c

    left_x_d = event.left_snapshot.x_d
    right_x_d = event.right_snapshot.x_d

    right_x = vcat(right_x_c, right_x_d)

    # With an ANN in front of the FMU, the snapshots hold the FMU states `z = g(x)`, not the
    # ODE states. Then, the reset `z_r = R(z_l)` is sampled in FMU coordinates
    # (`modifiedState = false`), see `modifiedStateEventSensitivity` for the ODE coordinates.
    modifiedState = nfmu.modifiedState
    if modifiedState
        # The right snapshot holds `g(x_right)`, which matches the FMU reset only up to the
        # tolerance of the state search, so recompute the reference the same way as the
        # perturbed resets below.
        FMIBase.apply!(c, event.left_snapshot)
        right_x = stateChange!(
            nfmu,
            c,
            vcat(left_x_c, left_x_d),
            event.t,
            event.indicator;
            modifiedState = false,
        )
    end

    @debug "sampleStateChangeJacobian(x = $(left_x_c))"

    c.solution.evals_∂xr_∂xl += 1

    numStates = length(left_x_c) + length(left_x_d)
    numConStates = length(left_x_c)

    jac = zeros(numStates, numStates)

    if nfmu.fmu.isDummyDiscrete
        jac[numStates, numStates] = 1.0
    end

    # # [ToDo] ONLY A TEST
    # new_left_x = copy(left_x)
    # if length(c.solution.snapshots) > 0 # c.t != t 
    #     sn = getPreviousSnapshot(c.solution, t; index=index)

    #     # this is the case for t = t_0
    #     if isnothing(sn)
    #         sn = getSnapshot(c.solution, t; index=index)
    #     end

    #     FMIBase.apply!(c, sn; x_c = new_left_x, t = t)
    #     #@info "[?] Set snapshot @ t=$(t) (sn.t=$(sn.t))"
    # end
    # new_right_x = stateChange!(nfmu, c, new_left_x, t, idx; snapshots = false, index=index)
    # statesChanged = (c.eventInfo.valuesOfContinuousStatesChanged == fmi2True)

    # # [ToDo: these tests should be included, but will drastically fail on FMUs with no support for get/setState]
    # # @assert statesChanged "Can't reproduce event (statesChanged)!" 
    # # @assert left_x == new_left_x "Can't reproduce event (left_x)!"
    # # @assert right_x == new_right_x "Can't reproduce event (right_x)!"

    at_least_one_state_change = false

    for i = 1:numConStates

        new_left_x_c = copy(left_x_c)
        new_left_x_c[i] += step

        # first, jump to before the event instance
        FMIBase.apply!(c, event.left_snapshot; x_c = new_left_x_c)

        # check event
        new_right_x = stateChange!(
            nfmu,
            c,
            vcat(new_left_x_c, left_x_d),
            event.t,
            event.indicator;
            modifiedState = false,
        )

        statesChanged = (c.eventInfo.valuesOfContinuousStatesChanged == fmi2True)
        at_least_one_state_change = statesChanged || at_least_one_state_change

        # if nfmu.fmu.isDummyDiscrete
        #     new_right_x = new_right_x[1:end-1]
        # end

        grad = (new_right_x .- right_x) ./ step

        # choose other direction
        if !statesChanged

            new_left_x_c = copy(left_x_c)
            new_left_x_c[i] -= step

            FMIBase.apply!(c, event.left_snapshot; x_c = new_left_x_c)

            new_right_x = stateChange!(
                nfmu,
                c,
                vcat(new_left_x_c, left_x_d),
                event.t,
                event.indicator;
                modifiedState = false,
            )

            statesChanged = (c.eventInfo.valuesOfContinuousStatesChanged == fmi2True)

            at_least_one_state_change = statesChanged || at_least_one_state_change

            # if nfmu.fmu.isDummyDiscrete
            #     new_right_x = new_right_x[1:end-1]
            # end

            if statesChanged
                grad = (new_right_x .- right_x) ./ -step
            else
                grad = (right_x .- right_x) # ... so zero, this state is not sensitive at all!
            end

        end

        jac[:, i] = grad
    end

    #@assert at_least_one_state_change "Sampling state change jacobian failed, can't find another state that triggers the event!"
    if !at_least_one_state_change
        @info "Sampling state change jacobian failed, can't find another state that triggers the event!\ncommon reasons for that are:\n(a) The FMU is not able to revisit events (which should be possible with fmiXGet/SetState).\n(b) The state change is not dependent on the previous state (hard reset).\nThis is printed only 3 times." maxlog =
            3
    end

    # finally, jump back to the correct FMU state
    FMIBase.apply!(c, event.right_snapshot)

    return jac
end

function sampleEventIndicatorValue(nfmu, c, event::FMUEvent, x, t)
    FMIBase.apply!(c, event.left_snapshot; x_c = collect(continuousStateView(c, x)), t = t)
    return conditionSingle(nfmu, c, event.indicator, x, t, nothing)
end

function sampleEventIndicatorGradient(nfmu, c, event::FMUEvent; step = 1e-8)
    left_x = vcat(event.left_snapshot.x_c, event.left_snapshot.x_d)
    grad = zeros(length(left_x))

    # Saltation needs the normal of the event surface. FMI exposes event
    # indicators as scalar functions, so we sample only the continuous-state
    # part and leave the dummy discrete coordinate insensitive.
    for i = 1:continuousStateLength(c)
        xp = copy(left_x)
        xm = copy(left_x)
        xp[i] += step
        xm[i] -= step

        fp = sampleEventIndicatorValue(nfmu, c, event, xp, event.t)
        fm = sampleEventIndicatorValue(nfmu, c, event, xm, event.t)
        grad[i] = (fp - fm) / (2step)
    end

    FMIBase.apply!(c, event.left_snapshot)
    return grad
end

function sampleEventIndicatorTimeGradient(nfmu, c, event::FMUEvent; step = 1e-8)
    left_x = vcat(event.left_snapshot.x_c, event.left_snapshot.x_d)
    fp = sampleEventIndicatorValue(nfmu, c, event, left_x, event.t + step)
    fm = sampleEventIndicatorValue(nfmu, c, event, left_x, event.t - step)

    FMIBase.apply!(c, event.left_snapshot)
    return (fp - fm) / (2step)
end

function sampleEventDynamics(nfmu, c, snapshot::FMUSnapshot, x, t)
    FMIBase.apply!(c, snapshot; x_c = collect(continuousStateView(c, x)), t = t)
    return unsense_copy(evaluateModel(nfmu, c, x; p = unsense(nfmu.p), t = t, force = true))
end

function sampleStateTransitionJacobian(nfmu, c, event::FMUEvent; saltation::Bool = false)
    reset_jac = sampleStateChangeJacobian(nfmu, c, event)

    if !saltation || event.indicator == 0
        return reset_jac
    end

    left_x = vcat(event.left_snapshot.x_c, event.left_snapshot.x_d)
    right_x = vcat(event.right_snapshot.x_c, event.right_snapshot.x_d)
    normal = sampleEventIndicatorGradient(nfmu, c, event)
    normal_t = sampleEventIndicatorTimeGradient(nfmu, c, event)
    f_left = sampleEventDynamics(nfmu, c, event.left_snapshot, left_x, event.t)
    f_right = sampleEventDynamics(nfmu, c, event.right_snapshot, right_x, event.t)

    denom = normal_t + sum(normal .* f_left)

    if abs(denom) <= EVENT_SALTATION_DENOM_ATOL
        logWarning(
            nfmu.fmu,
            "Skipping saltation correction for event $(event.indicator) at t=$(event.t), because the event-surface crossing denominator is $(denom).",
            3,
        )
        return reset_jac
    end

    # Saltation corrects the fixed-time reset Jacobian by the time shift of the
    # root. This is only used for solver-discovered state events; direct calls to
    # `affectFMU!` keep the fixed-time reset derivative expected by unit tests.
    transition_jac = reset_jac + (f_right .- reset_jac * f_left) * transpose(normal) / denom

    FMIBase.apply!(c, event.right_snapshot)
    return transition_jac
end

function useSaltationForEventSensitivity(integrator)
    # ReverseDiffAdjoint already differentiates through the solver's root
    # localization. The splice inside `affectFMU!` should therefore provide the
    # fixed-time reset derivative only; enabling a saltation correction here
    # double-counts root-time sensitivity for the current SciML callback path.
    return false
end

function scalarEventIndex(idx)
    idx isa Number && return Int(idx)

    @assert !isempty(idx) "Received an empty VectorContinuousCallback event index."

    local_pos = findfirst(!iszero, idx)
    @assert !isnothing(local_pos) "Received a VectorContinuousCallback event index without an active entry."

    value = Int(idx[local_pos])
    if value > 1
        return value
    end

    # Newer SciML callback code may pass either a vector of active indicator
    # numbers, e.g. `[2]`, or a view into an activity/sign vector, e.g.
    # `view([0, 1], 2:2)`. In the latter case the value is only a flag, so
    # recover the original indicator position from the view's parent indices.
    if idx isa SubArray
        parent_index = parentindices(idx)[1]
        if parent_index isa AbstractUnitRange
            return first(parent_index) + Int(local_pos) - 1
        elseif parent_index isa AbstractVector
            return Int(parent_index[local_pos])
        end
    end

    return Int(local_pos)
end

function is_integrator_sensitive(integrator)
    return istracked(integrator.u) ||
           istracked(integrator.t) ||
           isdual(integrator.u) ||
           isdual(integrator.t)
end

function stateChange!(
    nfmu,
    c,
    left_x::AbstractArray{<:Float64},
    t::Float64,
    idx;
    snapshots = nfmu.snapshots,
    modifiedState::Bool = nfmu.modifiedState, # `false` returns the new state in FMU coordinates
)

    fmi2EnterEventMode(c)
    handleEvents(c)

    right_x = left_x

    if isTrue(c.eventInfo.valuesOfContinuousStatesChanged)

        ignore_derivatives() do
            if idx == 0
                @debug "stateChange!($(idx)): NeuralFMU time event with state change.\nt = $(t)\nleft_x = $(left_x)"
            else
                @debug "stateChange!($(idx)): NeuralFMU state event with state change by indicator $(idx).\nt = $(t)\nleft_x = $(left_x)"
            end
        end

        right_x_fmu = getContinuousStates(c) # the new FMU state after handled events

        # if there is an ANN in front of the FMU, propagate the new FMU state back through it
        if modifiedState
            # The ODE state differs from the FMU state, because an ANN maps the ODE state
            # onto the FMU state. Search the ODE state `right_x` that the ANN maps onto the
            # new FMU state `right_x_fmu`; only the continuous states are searched, a dummy
            # discrete state is passed through.
            nx = length(right_x_fmu)
            before = fmi2GetEventIndicators(c)

            right_x, residual = solveModifiedState(nfmu, c, right_x_fmu, left_x[1:nx], t)

            @debug "stateChange!($(idx)): Propagated FMU state $(right_x_fmu) to ODE state $(right_x), residual $(residual)."

            if residual > MODIFIED_STATE_WARN_RTOL * (1.0 + norm(right_x_fmu))
                logWarning(
                    nfmu.fmu,
                    "stateChange!(...): The ANN in front of the FMU can't reproduce the FMU state after the event at t=$(t) (residual $(residual)), continuing with the closest state.\nRight state: $(right_x)\nRight FMU state: $(right_x_fmu)",
                    3,
                )
            end

            # time events (`idx == 0`) have no triggering indicator
            if idx > 0
                after = fmi2GetEventIndicators(c)
                if sign(before[idx]) != sign(after[idx])
                    logError(
                        nfmu.fmu,
                        "Eventhandling failed,\nRight state: $(right_x)\nRight FMU state: $(right_x_fmu)\nIndicator (bef.): $(before[idx])\nIndicator (aft.): $(after[idx])",
                    )
                end
            end

        else # if there is no ANN above, then:
            right_x = right_x_fmu
        end

        if c.fmu.isDummyDiscrete
            x_d = [left_x[end]+1] # [round(Integer, left_x[end]+1)]
            right_x = vcat(right_x, x_d)
            # @assert right_x_fmu[end] == x_d+1 "dummy discrete state missmatch after event, before $(x_d), after $(right_x_fmu[end]), should be $(x_d+1)."
        end

    else

        if c.fmu.isDummyDiscrete
            x_d = [left_x[end]+1] # [round(Integer, left_x[end]+1)]
            right_x = vcat(right_x[1:(end-1)], x_d)
        end

        ignore_derivatives() do
            if idx == 0
                @debug "stateChange!($(idx)): NeuralFMU time event without state change.\nt = $(t)\nx = $(left_x)"
            else
                @debug "stateChange!($(idx)): NeuralFMU state event without state change by indicator $(idx).\nt = $(t)\nx = $(left_x)"
            end
        end

    end

    @assert length(right_x) == length(left_x) "len mismatch $(length(right_x)) != $(length(left_x))"

    return right_x
end

# Handles the upcoming event
function affectFMU!(nfmu::ME_NeuralFMU, c::FMU2Component, integrator, idx)

    @assert isContinuousTimeMode(c) "affectFMU!(...):\n" * ERR_MSG_CONT_TIME_MODE

    @assert getCurrentInstance(nfmu.fmu) == c "Thread `$(Threads.threadid())` wants to evaluate wrong component!"
    # assert_integrator_valid(integrator)

    idx = scalarEventIndex(idx)

    if c.termSim
        @warn "termSim requested!" maxlog=3
        #logError(c.fmu, "affectFMU!(...): Terminating simulation because of previous errors.")
        #terminate!(integrator)
        #return
    end

    # [NOTE] Here unsensing is OK, because we just want to reset the FMU to the correct state!
    #        The values come directly from the integrator and are NOT function arguments!
    t = unsense(integrator.t)
    left_x = unsense_copy(integrator.u)
    right_x = nothing

    @debug "affectFMU!(t=$(t), ...) -> ..."

    ignore_derivatives() do

        # if snapshots && length(c.solution.snapshots) > 0 
        #     sn = getSnapshot(c.solution, t)
        #     FMIBase.apply!(c, sn)
        # end

        #if c.x != left_x
        # capture status of `force`    


        # there are fx-evaluations before the event is handled, reset the FMU state to the current integrator step
        evaluateModel(
            nfmu,
            c,
            left_x;
            p = callbackParameters(nfmu, integrator),
            t = t,
            force = true,
        ) # evaluate NeuralFMU (set new states)
        # [NOTE] No need to reset time here, because we did pass a event instance! 
        #        c.default_t = -1.0

        #end
    end

    left_snapshot = nothing
    right_snapshot = nothing

    if nfmu.snapshots
        left_snapshot = snapshot!(c)
        if c.fmu.isDummyDiscrete
            left_snapshot.x_d = round.(left_snapshot.x_d; digits = 0)
        end
        #@info "left! $(length(c.solution.snapshots))"
        push!(c.solution.snapshots, left_snapshot)
    end

    # INFO: entereventmode and handleevents happens inside stateChange!
    right_x = stateChange!(nfmu, c, left_x, t, idx)
    continuousStateChanged = isTrue(c.eventInfo.valuesOfContinuousStatesChanged)
    continuousNoop = continuousEventIsNoop(c, left_x, right_x)
    coalescedNoop = isNearCoincidentNoopEvent(c, t, left_x, right_x)
    sensitivityActive =
        continuousStateChanged && idx != -1 && !continuousNoop && !coalescedNoop

    if nfmu.fmu.isDummyDiscrete
        c.x_d = right_x[end:end]
    end

    if nfmu.snapshots
        right_snapshot = snapshot!(c)
        if c.fmu.isDummyDiscrete
            right_snapshot.x_d = round.(right_snapshot.x_d; digits = 0)
        end
        push!(c.solution.snapshots, right_snapshot)
    end

    # log event 
    event = nothing
    ignore_derivatives() do
        if idx != -1
            _left_x = left_x
            _right_x = isnothing(right_x) ? _left_x : unsense_copy(right_x)

            #@assert c.eventInfo.valuesOfContinuousStatesChanged == (_left_x != _right_x) "FMU says valuesOfContinuousStatesChanged $(c.eventInfo.valuesOfContinuousStatesChanged), but states say different!"
            event = FMUEvent(unsense(t), UInt64(idx), _left_x, _right_x)

            event.left_snapshot = left_snapshot
            event.right_snapshot = right_snapshot

            if continuousNoop
                @debug "affectFMU!(...): Event $(idx) at t=$(t) is a continuous-state no-op for sensitivity propagation."
            end
            if coalescedNoop
                @debug "affectFMU!(...): Event $(idx) at t=$(t) is coalesced with the previous near-coincident no-op event."
            end

            push!(c.solution.events, event)
        end

        # calculates state events per second
        pt = t - nfmu.tspan[1]
        ne = 0
        for event in c.solution.events
            #if t - event.t < pt 
            if event.indicator > 0 # count only state events
                ne += 1
            end
            #end
        end
        ratio = ne / pt

        if ne >= 100 && ratio > c.fmu.executionConfig.maxStateEventsPerSecond
            logError(
                c.fmu,
                "Event chattering detected $(round(Integer, ratio)) state events/s (allowed are $(c.fmu.executionConfig.maxStateEventsPerSecond)), aborting at t=$(t) (rel. t=$(pt)) at state event $(ne):",
            )
            for i = 1:c.fmu.modelDescription.numberOfEventIndicators
                num = 0
                for e in c.solution.events
                    if e.indicator == i
                        num += 1
                    end
                end
                if num > 0
                    logError(
                        c.fmu,
                        "\tEvent indicator #$(i) triggered $(num) ($(round(num/ne*100.0; digits=1))%)",
                    )
                end
            end

            terminate!(integrator)
        end
    end

    # `valuesOfContinuousStatesChanged` comes from FMI event handling. For AD we
    # also require a numerically visible continuous-state reset; some FMUs emit a
    # second event a few ulps later that only updates internal/discrete state.
    # Those duplicate no-op events are still logged, but they must not introduce
    # an extra adjoint transition.
    if continuousStateChanged

        # [ToDo] Do something with that information, e.g. use for FiniteDiff sampling step size determination and pass to ODE solver!
        c.x_nominals = fmi2GetNominalsOfContinuousStates(c)

        # sensitivities needed
        if nfmu.snapshots && sensitivityActive

            # Here, we compute a transformation that leads to the new state, and apply it.
            # This is a workaround as substitute for giving frule/rrule here.

            # (with an ANN in front of the FMU, `jac` is the reset in FMU coordinates,
            # the saltation correction is not implemented for this case)
            jac = sampleStateTransitionJacobian(
                nfmu,
                c,
                event;
                saltation = !nfmu.modifiedState &&
                            useSaltationForEventSensitivity(integrator),
            )

            if eventTransitionJacobianIsUsable(jac)
                if nfmu.modifiedState
                    integrator.u[:] =
                        modifiedStateEventSensitivity(nfmu, c, integrator, right_x, jac, t)
                else
                    VJP = jac * integrator.u
                    #tgrad = tvec .* integrator.t
                    staticOff = right_x .- unsense(VJP) # .- unsense(tgrad)

                    #@info "$(unsense(jac))"

                    # [ToDo] add (sampled) time gradient
                    integrator.u[:] = staticOff + VJP # + tgrad
                end
            else
                logWarning(
                    nfmu.fmu,
                    "Skipping sampled event sensitivity for event $(event.indicator) at t=$(event.t), because the reset Jacobian is non-finite or exceeds $(EVENT_SENSITIVITY_JACOBIAN_MAX).",
                    3,
                )
                integrator.u[:] = right_x
            end
        else
            # If the event is a no-op for continuous states, near-coincident with
            # a previous no-op, or lacks snapshots, apply the value update without
            # synthesizing an additional event Jacobian.
            integrator.u[:] = right_x
        end
    else
        if c.fmu.isDummyDiscrete
            integrator.u[end] = integrator.u[end] + (right_x[end]-integrator.u[end])
        else
            # if no cont. state change and no dummy discrete state, then nothing to do here.
        end
    end

    # if c.fmu.isDummyDiscrete
    #     integrator.u[end] = unsense(right_x[end])
    # end


    # [Note] enabling this causes serious issues with time events! (wrong sensitivities!)
    #u_modified!(integrator, true)

    c.solution.evals_affect += 1

    return nothing
end

# function condition!(
#     nfmu::ME_NeuralFMU,
#     c::FMU2Component,
#     out,
#     x,
#     t,
#     integrator,
#     handleEventIndicators,
# )

#     # prev_t = c.default_t
#     # prev_ec = c.default_ec
#     # prev_ec_idcs = c.default_ec_idcs

#     # c.default_t = t
#     # c.default_ec = zeros(Float64, 0) # out 
#     # c.default_ec_idcs = handleEventIndicators

#     # evaluateModel(nfmu, c, x; t=t)

#     # # reset
#     # c.default_t = prev_t
#     # c.default_ec = prev_ec
#     # c.default_ec_idcs = prev_ec_idcs

#     out[:] = [x[1]-0.0, x[1]-0.0]

#     # c.solution.evals_condition += 1

#     # JVP = [1 0 0; 1 0 0] * x #c.∂e_∂x.mtx * x
#     # C = unsense(c.output.ec) - unsense(JVP)
#     # out[:] = JVP + C

#     # return nothing
# end

# function affectFMU!(nfmu::ME_NeuralFMU, c::FMU2Component, integrator, idx)

#     # t = unsense(integrator.t)
#     # left_x = unsense_copy(integrator.u)
#     # right_x = nothing

#     # evaluateModel(nfmu, c, left_x; t = t, force=true) # evaluate NeuralFMU (set new states)

#     # right_x = stateChange!(nfmu, c, left_x, t, idx)

#     # jac = [0.0 0.0 0.0; 0.0 -0.7 0.0; 0.0 0.0 1.0]
#     # JVP = jac * integrator.u
#     # C = right_x .- unsense(JVP) # .- unsense(tgrad)

#     # integrator.u[:] = C + JVP 

#     # c.solution.evals_affect += 1

#     out = zeros(2)
#     condition!(nfmu, c, out, unsense(integrator.u), unsense(integrator.t), integrator, [1,2]) 

#     if idx == 2
#         return 
#     end

#     if out[idx] < 0.0
#         integrator.u[1] = 1e-10
#         integrator.u[2] = integrator.u[2]*-0.7

#         if nfmu.fmu.isDummyDiscrete
#             integrator.u[3] = round(Integer, integrator.u[3]+1)
#         end
#     end

#     return nothing
# end

# Does one step in the simulation.
function stepCompleted(
    nfmu::ME_NeuralFMU,
    c::FMU2Component,
    x,
    t,
    integrator,
    tStart,
    tStop,
)

    @assert isContinuousTimeMode(c) "stepCompleted(...):\n" * ERR_MSG_CONT_TIME_MODE

    if c.termSim
        #logError(c.fmu, "stepCompleted(...): Terminating simulation because of previous errors.")
        #terminate!(integrator)
        #return
    end

    # [Note] enabling this causes serious issues with time events! (wrong sensitivities!)
    # u_modified!(integrator, false)

    c.solution.evals_stepcompleted += 1

    # if snapshots
    #     FMIBase.snapshot!(c.solution)
    # end

    if !isnothing(c.progressMeter)
        t = unsense(t)
        dt = unsense(integrator.t) - unsense(integrator.tprev)
        events = length(c.solution.events)
        steps = c.solution.evals_stepcompleted
        simLen = tStop - tStart
        c.progressMeter.desc = "t=$(roundToLength(t, 10))s | Δt=$(roundToLength(dt, 10))s | STPs=$(steps) | EVTs=$(events) |"
        #@info "$(tStart)   $(tStop)   $(t)"

        if simLen > 0.0
            ProgressMeter.update!(
                c.progressMeter,
                floor(Integer, 1000.0 * (t - tStart) / simLen),
            )
        end
    end

    status, enterEventMode, terminateSimulation = fmi2CompletedIntegratorStep(c, fmi2False)

    if isTrue(terminateSimulation)
        logError(c.fmu, "stepCompleted(...): FMU requested termination!")
        terminate!(integrator)
    end

    @debug "stepCompleted(t=$(t), ...) -> enterEventMode=$(c.enterEventMode), terminateSimulation=$(c.terminateSimulation)"

    if isTrue(enterEventMode)
        affectFMU!(nfmu, c, integrator, -1)
    else
        # ToDo: it would be sufficient to only update inputs here, 
        # however, this is not that easy for generic NFMUs 
        # Info: We use c.x and c.t, so that no state update happens. 
        # This is important, only inputs are allowed to be set here!
        #x = c.x 
        #t = c.t
        # if c.fmu.isDummyDiscrete
        #     x = vcat(x, c.x_d)
        # end
        evaluateModel(nfmu, c, x; p = callbackParameters(nfmu, integrator), t = t)
    end

    #end

    # assert_integrator_valid(integrator)
end

# function stepCompleted(
#     nfmu::ME_NeuralFMU,
#     c::FMU2Component,
#     x,
#     t,
#     integrator,
#     tStart,
#     tStop,
# )

# end

# [ToDo] (1) This must be in-place 
#        (2) getReal must be replaced with the inplace getter within c(...)
#        (3) remove unsense to determine save value sensitivities
# save FMU values 
function saveValues(nfmu::ME_NeuralFMU, c::FMU2Component, recordValues, _x, _t, integrator)

    @assert isContinuousTimeMode(c) "saveValues(...):\n" * ERR_MSG_CONT_TIME_MODE

    t = unsense(_t)
    x = unsense(_x)

    c.solution.evals_savevalues += 1

    # ToDo: Evaluate on light-weight model (sub-model) without fmi2GetXXX or similar and the bottom ANN
    # (`x` and `t` are unsensed here, so the parameters are unsensed as well)
    p = unsense(callbackParameters(nfmu, integrator))
    evaluateModel(nfmu, c, x; p = p, t = t) # evaluate NeuralFMU (set new states)

    values = fmi2GetReal(c, recordValues)

    @debug "Save values @t=$(t)\nintegrator.t=$(unsense(integrator.t))\n$(values)"

    return (values...,)
end

function saveEigenvalues(
    nfmu::ME_NeuralFMU,
    c::FMU2Component,
    _x,
    _t,
    integrator,
    sensitivity::Symbol,
)

    @assert c.state == fmi2ComponentStateContinuousTimeMode "saveEigenvalues(...):\n" *
                                                            FMIBase.ERR_MSG_CONT_TIME_MODE

    c.solution.evals_saveeigenvalues += 1

    p = callbackParameters(nfmu, integrator)

    A = nothing
    if sensitivity == :ForwardDiff
        A = ForwardDiff.jacobian(x -> evaluateModel(nfmu, c, x; p = p, t = _t), _x) # TODO: chunk_size!, remove y from dx_y
    elseif sensitivity == :ReverseDiff
        A = ReverseDiff.jacobian(x -> evaluateModel(nfmu, c, x; p = p, t = _t), _x)
    elseif sensitivity == :Zygote
        A = Zygote.jacobian(x -> evaluateModel(nfmu, c, x; p = p, t = _t), _x)[1]
    elseif sensitivity == :none
        A = ForwardDiff.jacobian(x -> evaluateModel(nfmu, c, x; p = p, t = _t), unsense(_x))
    end
    eigs, _ = DifferentiableEigen.eigen(A)

    return (eigs...,)
end

function fx(
    nfmu::ME_NeuralFMU,
    c::FMU2Component,
    dx,#::Array{<:Real},
    x,#::Array{<:Real},
    p,#::Array,
    t,
)#::Real) 

    ignore_derivatives() do
        c.solution.evals_fx_inplace += 1
    end
    @debug "f(t=$(t), ...) [in-place, eval count: $(c.solution.evals_fx_inplace)]"

    if isnothing(c)
        # this should never happen!
        @warn "fx() called without allocated FMU instance!"
        return zeros(length(x))
    end

    ############

    evaluateModel(nfmu, c, dx, x; p = p, t = t)

    return dx
end

function fx(
    nfmu::ME_NeuralFMU,
    c::FMU2Component,
    x,#::Array{<:Real},
    p,#::Array,
    t,
)#::Real) 
    ignore_derivatives() do
        c.solution.evals_fx_outofplace += 1
    end
    @debug "f(t=$(t), ...) [out-of-place, eval count: $(c.solution.evals_fx_outofplace)]"

    if c === nothing
        # this should never happen!
        return zeros(length(x))
    end

    return evaluateModel(nfmu, c, x; p = p, t = t)
end

##### EVENT HANDLING END

"""
Constructs a ME-NeuralFMU where the FMU is at an arbitrary location inside of the NN.

# Arguments
    - `fmu` the considered FMU inside the NN 
    - `model` the NN topology (e.g. Flux.chain)
    - `tspan` simulation time span
    - `solver` an ODE Solver (default=`nothing`, heurisitically determine one) 

# Keyword arguments
    - `recordValues` additionally records internal FMU variables 
"""
function ME_NeuralFMU(
    fmu::FMU2,
    model,
    tspan,
    solver = nothing;
    recordValues = nothing,
    saveat = nothing,
    solvekwargs...,
)

    if !is64(model)
        model = convert64(model)
        logInfo(
            fmu,
            "Model is not Float64, but this is required for (neural) FMUs.\nModel parameters are automatically converted to Float64.",
        )
    end

    p, re = FMIFlux.destructure(model)
    nfmu = ME_NeuralFMU{typeof(model),typeof(re)}(model, p, re)

    ######

    nfmu.fmu = fmu

    nfmu.saved_values = nothing

    nfmu.recordValues = prepareValueReference(fmu, recordValues)

    nfmu.tspan = tspan
    nfmu.solver = solver
    nfmu.saveat = saveat
    nfmu.solvekwargs = solvekwargs
    nfmu.parameters = nothing

    ######

    nfmu
end

"""
Constructs a CS-NeuralFMU where the FMU is at an arbitrary location inside of the ANN.

# Arguents
    - `fmu` the considered FMU inside the ANN 
    - `model` the ANN topology (e.g. Flux.Chain)
    - `tspan` simulation time span

# Keyword arguments
    - `recordValues` additionally records FMU variables 
"""
function CS_NeuralFMU(fmu::FMU2, model, tspan; recordValues = [])

    if !is64(model)
        model = convert64(model)
        logInfo(
            fmu,
            "Model is not Float64, but this is necessary for (Neural)FMUs.\nModel parameters are automatically converted to Float64.",
        )
    end

    nfmu = CS_NeuralFMU{FMU2,FMU2Component}()

    nfmu.fmu = fmu
    nfmu.model = model
    nfmu.tspan = tspan

    FMIFlux.params(nfmu; destructure = true)

    return nfmu
end

function CS_NeuralFMU(fmus::Vector{<:FMU2}, model, tspan; recordValues = [])

    if !is64(model)
        model = convert64(model)
        for fmu in fmus
            logInfo(
                fmu,
                "Model is not Float64, but this is necessary for (Neural)FMUs.\nModel parameters are automatically converted to Float64.",
            )
        end
    end

    nfmu = CS_NeuralFMU{Vector{FMU2},Vector{FMU2Component}}()

    nfmu.fmu = fmus
    nfmu.model = model
    nfmu.tspan = tspan

    FMIFlux.params(nfmu; destructure = true)

    return nfmu
end

function checkExecTime(integrator, nfmu::ME_NeuralFMU, c, max_execution_duration::Real)
    dist = max(nfmu.execution_start + max_execution_duration - time(), 0.0)

    if dist <= 0.0
        logInfo(
            nfmu.fmu,
            "Reached max execution duration ($(max_execution_duration)), terminating integration ...",
        )
        terminate!(integrator)
    end

    return 1.0
end

function getInstance(nfmu::NeuralFMU)
    return hasCurrentInstance(nfmu.fmu) ? getCurrentInstance(nfmu.fmu) : nothing
end

# ToDo: Separate this: NeuralFMU creation and solving!
"""
    
    nfmu(x_start, tspan; kwargs)

Evaluates the ME_NeuralFMU `nfmu` in the timespan given during construction or in a custom timespan from `t_start` to `t_stop` for a given start state `x_start`.

# Keyword arguments

[ToDo]
"""
function (nfmu::ME_NeuralFMU)(
    x_start::Union{Vector{Float64},Nothing} = nfmu.x0, # Union{Array{<:Real},Nothing}
    tspan::Tuple{Float64,Float64} = nfmu.tspan;
    showProgress::Bool = false,
    progressDescr::String = DEFAULT_PROGRESS_DESCR,
    tolerance::Union{Float64,Nothing} = nothing,
    parameters::Union{Dict{<:Any,<:Any},Nothing} = nothing,
    p = nfmu.p,
    alg = nfmu.solver,
    solver = nothing,
    saveEventPositions::Bool = false,
    max_execution_duration::Float64 = -1.0,
    recordValues::fmi2ValueReferenceFormat = nfmu.recordValues,
    recordEigenvaluesSensitivity::Symbol = :none,
    recordEigenvalues::Bool = (recordEigenvaluesSensitivity != :none),
    saveat = nfmu.saveat, # ToDo: Data type 
    sensealg = nfmu.fmu.executionConfig.sensealg, # ToDo: AbstractSensitivityAlgorithm
    writeSnapshot::Union{FMUSnapshot,Nothing} = nothing,
    readSnapshot::Union{FMUSnapshot,Nothing} = nothing,
    cleanSnapshots::Bool = false,
    useStepCallback::Bool = true,
    useStartCallback::Bool = true,
    dummyDiscreteStateIfRequired::Bool = true,
    solvekwargs...,
)

    if istracked(p)
        # not required if we snapshot every step
        if !nfmu.fmu.isDummyDiscrete # && !nfmu.fmu.executionConfig.snapshot_every_step
            if dummyDiscreteStateIfRequired
                logInfo(
                    nfmu.fmu,
                    "Activating dummy discrete state for FMU to perform proper sensitivity analysis with ReverseDiff.jl.",
                )
                nfmu.fmu.isDummyDiscrete = true
            end
        end
    elseif isdual(p)
        if nfmu.fmu.isDummyDiscrete
            logInfo(
                nfmu.fmu,
                "Deactivating dummy discrete state for FMU to save performance for sensitivity analyis with ForwardDiff.jl.",
            )
            nfmu.fmu.isDummyDiscrete = false
        end
    else
        if nfmu.fmu.isDummyDiscrete
            logInfo(
                nfmu.fmu,
                "Deactivating dummy discrete state for FMU to save performance for simulation.",
            )
            nfmu.fmu.isDummyDiscrete = false
        end
    end

    if !isnothing(solver)
        alg = solver
        logWarning(
            nfmu.fmu,
            "Neural FMU called with keyword `solver`, please use `alg` instead.",
            3,
        )
    end

    if !isnothing(saveat)
        if saveat[1] != tspan[1] || saveat[end] != tspan[end]
            logWarning(
                nfmu.fmu,
                "NeuralFMU changed time interval, start time is $(tspan[1]) and stop time is $(tspan[end]), but saveat from constructor gives $(saveat[1]) and $(saveat[end]).\nPlease provide correct `saveat` via keyword with matching start/stop time.",
                1,
            )
            saveat = collect(saveat)
            while saveat[1] < tspan[1]
                popfirst!(saveat)
            end
            while saveat[end] > tspan[end]
                pop!(saveat)
            end
        end
    end

    recordValues = prepareValueReference(nfmu.fmu, recordValues)

    saving = (length(recordValues) > 0)

    solvekwargs = Dict{Symbol,Any}(solvekwargs...)
    tspan = setupSolver!(nfmu.fmu, tspan, solvekwargs)

    # if haskey(solvekwargs, :dtmax)
    #     @info "Removing automatically set dtmax=$(solvekwargs[:dtmax])" # maxlog=3
    #     delete!(solvekwargs, :dtmax)
    # end

    #@assert !haskey(solvekwargs, :dtmax) "dtmax"

    t_start = tspan[1]
    t_stop = tspan[end]

    nfmu.p = p

    ignore_derivatives() do

        nfmu.firstRun = true

        nfmu.tolerance = tolerance

        if isnothing(parameters)
            if !isnothing(nfmu.fmu.default_p_refs)
                nfmu.parameters =
                    Dict(nfmu.fmu.default_p_refs .=> unsense(nfmu.fmu.default_p))
            end
        else
            nfmu.parameters = parameters
        end

    end

    callbacks = []

    c = getInstance(nfmu)

    @assert !issense(x_start) "sense x_start not tested!"

    if nfmu.fmu.isDummyDiscrete
        x_start = vcat(x_start, 0.0)
    end
    # if nfmu.fmu.isDummyDiscrete
    #     buf = similar(x_start, length(x_start)+1)
    #     buf[1:end-1] = x_start
    #     buf[end] = 0.0
    #     if issense(x_start)
    #         @info "$(x_start[1].deriv) | $(buf[1].deriv)"
    #     end
    #     x_start = buf
    # end

    nfmu.tspan = tspan
    nfmu.x0 = x_start

    @debug "ME_NeuralFMU(showProgress=$(showProgress), tspan=$(tspan), x0=$(nfmu.x0))"

    if useStartCallback
        @debug "ME_NeuralFMU: Starting callback..."
        c = startCallback(nothing, nfmu, c, t_start, writeSnapshot, readSnapshot)
    end

    ignore_derivatives() do

        @debug "ME_NeuralFMU: Defining callbacks..."

        # custom callbacks
        for cb in nfmu.customCallbacksBefore
            push!(callbacks, cb)
        end

        nfmu.fmu.hasStateEvents = (c.fmu.modelDescription.numberOfEventIndicators > 0)
        nfmu.fmu.hasTimeEvents = (c.eventInfo.nextEventTimeDefined == fmi2True)

        # time event handling

        if nfmu.fmu.executionConfig.handleTimeEvents && nfmu.fmu.hasTimeEvents
            timeEventCb = IterativeCallback(
                (integrator) -> time_choice(nfmu, c, integrator, t_start, t_stop),
                (integrator) -> affectFMU!(nfmu, c, integrator, 0),
                Float64;
                initial_affect = (c.eventInfo.nextEventTime == t_start), # already checked in the outer closure: c.eventInfo.nextEventTimeDefined == fmi2True
                save_positions = (saveEventPositions, saveEventPositions),
            )

            push!(callbacks, timeEventCb)
        end

        # state event callback

        if c.fmu.hasStateEvents && c.fmu.executionConfig.handleStateEvents

            handleIndicators = nothing

            # if we want a specific subset
            if !isnothing(c.fmu.handleEventIndicators)
                handleIndicators = c.fmu.handleEventIndicators
            else # handle all
                handleIndicators = collect(
                    UInt32(i) for i = 1:c.fmu.modelDescription.numberOfEventIndicators
                )
            end

            numEventInds = length(handleIndicators)

            # Vector callbacks are the fast/default event path. If a
            # ReverseDiff gradient produces a non-finite or threshold-breaking
            # result, computeGradient! retries that one solve with this
            # task-local flag set so each event indicator is replayed through an
            # independent scalar callback.
            if c.fmu.executionConfig.useVectorCallbacks && !forceScalarEventCallbacks()

                # FMI event indicators can remain numerically close to zero
                # immediately after event handling. A root tolerance above
                # SciML's machine-epsilon default prevents a duplicate no-op
                # callback from entering the ReverseDiff adjoint path.
                eventCb = VectorContinuousCallback(
                    (out, x, t, integrator) ->
                        condition!(nfmu, c, out, x, t, integrator, handleIndicators),
                    (integrator, idx) -> affectFMU!(nfmu, c, integrator, idx),
                    numEventInds;
                    rootfind = RightRootFind,
                    abstol = EVENT_SENSITIVITY_TIME_ATOL,
                    reltol = EVENT_SENSITIVITY_TIME_ATOL,
                    save_positions = (saveEventPositions, saveEventPositions),
                    interp_points = c.fmu.executionConfig.rootSearchInterpolationPoints,
                )
                push!(callbacks, eventCb)
            else

                for idx = 1:c.fmu.modelDescription.numberOfEventIndicators
                    # See the vector-callback branch above for why this event
                    # root tolerance is intentionally shared by scalar callbacks.
                    eventCb = ContinuousCallback(
                        (x, t, integrator) ->
                            conditionSingle(nfmu, c, idx, x, t, integrator),
                        (integrator) -> affectFMU!(nfmu, c, integrator, idx);
                        rootfind = RightRootFind,
                        abstol = EVENT_SENSITIVITY_TIME_ATOL,
                        reltol = EVENT_SENSITIVITY_TIME_ATOL,
                        save_positions = (saveEventPositions, saveEventPositions),
                        interp_points = c.fmu.executionConfig.rootSearchInterpolationPoints,
                    )
                    push!(callbacks, eventCb)
                end
            end
        end

        if max_execution_duration > 0.0
            terminateCb = ContinuousCallback(
                (x, t, integrator) ->
                    checkExecTime(integrator, nfmu, c, max_execution_duration),
                (integrator) -> terminate!(integrator);
                save_positions = (false, false),
            )
            push!(callbacks, terminateCb)
            logInfo(nfmu.fmu, "Setting max execeution time to $(max_execution_duration)")
        end

        if showProgress
            c.progressMeter =
                ProgressMeter.Progress(1000; desc = progressDescr, color = :blue, dt = 1.0)
            ProgressMeter.update!(c.progressMeter, 0) # show it!
        else
            c.progressMeter = nothing
        end

        # integrator step callback
        if useStepCallback
            stepCb = FunctionCallingCallback(
                (x, t, integrator) ->
                    stepCompleted(nfmu, c, x, t, integrator, t_start, t_stop);
                func_everystep = true,
                func_start = true,
            )
            push!(callbacks, stepCb)
        end

        # [ToDo] Allow for AD-primitives for sensitivity analysis of recorded values
        if saving
            c.solution.values = SavedValues(
                Float64,
                Tuple{collect(Float64 for i = 1:length(recordValues))...},
            )
            c.solution.valueReferences = recordValues

            if isnothing(saveat)
                savingCB = SavingCallback(
                    (x, t, integrator) ->
                        saveValues(nfmu, c, recordValues, x, t, integrator),
                    c.solution.values,
                )
            else
                savingCB = SavingCallback(
                    (x, t, integrator) ->
                        saveValues(nfmu, c, recordValues, x, t, integrator),
                    c.solution.values,
                    saveat = saveat,
                )
            end
            push!(callbacks, savingCB)
        end

        if recordEigenvalues

            @assert recordEigenvaluesSensitivity ∈
                    (:none, :ForwardDiff, :ReverseDiff, :Zygote) "Keyword `recordEigenvaluesSensitivity` must be one of (:none, :ForwardDiff, :ReverseDiff, :Zygote)"

            recordEigenvaluesType = nothing
            if recordEigenvaluesSensitivity == :ForwardDiff
                recordEigenvaluesType = FMISensitivity.ForwardDiff.Dual
            elseif recordEigenvaluesSensitivity == :ReverseDiff
                recordEigenvaluesType = FMISensitivity.ReverseDiff.TrackedReal
            elseif recordEigenvaluesSensitivity ∈ (:none, :Zygote)
                recordEigenvaluesType = fmi2Real
            end

            dtypes = collect(
                recordEigenvaluesType for
                _ = 1:(2*length(c.fmu.modelDescription.stateValueReferences))
            )
            c.solution.eigenvalues = SavedValues(recordEigenvaluesType, Tuple{dtypes...})

            savingCB = nothing
            if isnothing(saveat)
                savingCB = SavingCallback(
                    (u, t, integrator) -> saveEigenvalues(
                        nfmu,
                        c,
                        u,
                        t,
                        integrator,
                        recordEigenvaluesSensitivity,
                    ),
                    c.solution.eigenvalues,
                )
            else
                savingCB = SavingCallback(
                    (u, t, integrator) -> saveEigenvalues(
                        nfmu,
                        c,
                        u,
                        t,
                        integrator,
                        recordEigenvaluesSensitivity,
                    ),
                    c.solution.eigenvalues,
                    saveat = saveat,
                )
            end
            push!(callbacks, savingCB)
        end

    end # ignore_derivatives

    # custom callbacks
    for cb in nfmu.customCallbacksAfter
        push!(callbacks, cb)
    end

    prob = nothing

    function fx_ip(dx, x, p, t)
        fx(nfmu, c, dx, x, p, t)
        return nothing
    end

    # function fx_op(x, p, t)
    #     return fx(nfmu, c, x, p, t)
    # end

    # function fx_jac(J, x, p, t)
    #     J[:] = ReverseDiff.jacobian(_x -> fx_op(_x, p, t), x)
    #     return nothing 
    # end

    # function jvp(Jv, v, x, p, t)
    #     n = length(x)
    #     J = similar(x, (n, n))
    #     fx_jac(J, x, p, t)
    #     Jv[:] = J * v 
    #     return nothing
    # end

    # function vjp(Jv, v, x, p, t)
    #     n = length(x)
    #     J = similar(x, (n, n))
    #     fx_jac(J, x, p, t)
    #     Jv[:] = v' * J
    #     return nothing
    # end

    ff = ODEFunction{true}(fx_ip) # ; jvp=jvp, vjp=vjp, jac=fx_jac) # tgrad=nothing
    prob = ODEProblem{true}(ff, nfmu.x0, nfmu.tspan, p)

    # [TODO] that (using ReverseDiffAdjoint) should work now with `autodiff=false` 
    if isnothing(sensealg)
        #if isnothing(solver)

        #     logWarning(nfmu.fmu, "No solver keyword detected for NeuralFMU.\nOnly relevant if you use AD: Continuous adjoint method is applied, which requires solving backward in time.\nThis might be not supported by every FMU.", 1)
        #     sensealg = InterpolatingAdjoint(; autojacvec=ReverseDiffVJP(true), checkpointing=true)
        # elseif isimplicit(solver)
        #     @assert !(alg_autodiff(solver) isa AutoForwardDiff) "Implicit solver using `autodiff=true` detected for NeuralFMU.\nThis is currently not supported, please use `autodiff=false` as solver keyword.\nExample: `Rosenbrock23(autodiff=false)` instead of `Rosenbrock23()`."

        #     logWarning(nfmu.fmu, "Implicit solver detected for NeuralFMU.\nOnly relevant if you use AD: Continuous adjoint method is applied, which requires solving backward in time.\nThis might be not supported by every FMU.", 1)
        #     sensealg = InterpolatingAdjoint(; autojacvec=ReverseDiffVJP(true), checkpointing=true)
        # else
        sensealg = ReverseDiffAdjoint()
        #end
    end

    args = Vector{Any}()
    kwargs = Dict{Symbol,Any}(nfmu.solvekwargs..., solvekwargs...)

    if !isnothing(saveat)
        kwargs[:saveat] = saveat
    end

    ignore_derivatives() do
        # `alg` holds the solver from the constructor (`nfmu.solver`), the `alg` keyword or
        # the deprecated `solver` keyword (see above).
        if !isnothing(alg)
            push!(args, alg)
        end
    end

    #kwargs[:callback]=CallbackSet(callbacks...)
    #kwargs[:sensealg]=sensealg
    #kwargs[:u0] = nfmu.x0 # this is because of `IntervalNonlinearProblem has no field u0`

    @debug "Start solving ... tspan=$(nfmu.tspan)\nx0: $(nfmu.x0)\nargs: $(args...)\nkwargs: $(kwargs...)"

    if isdual(p)
        if !haskey(kwargs, :dt)
            kwargs[:dt] = 1e-6
            @warn "ForwardDiff currently requires setting `dt` to work, but is not set. Automatically setting dt=$(kwargs[:dt]).\nThis is printed only 3 times." maxlog=3
        end
    end

    #@info "kwargs: $(kwargs)"

    # for callback in callbacks
    #     @info first("$(callback)", 128)
    # end

    c.solution.states = solve(
        prob,
        args...;
        callback = CallbackSet(callbacks...),
        sensealg = sensealg,
        u0 = nfmu.x0,
        kwargs...,
    )

    @debug "Finished solving!"

    ignore_derivatives() do

        @assert !isnothing(c.solution.states) "Solving NeuralODE returned `nothing`!"

        # # ReverseDiff returns an array instead of an ODESolution, this needs to be corrected
        # # [TODO] doesn`t Array cover the TrackedArray case?
        if isa(c.solution.states, TrackedArray) || isa(c.solution.states, Array)

            @assert !isnothing(saveat) "Keyword `saveat` is nothing, please provide the keyword when using ReverseDiff."

            ts = collect(saveat)
            # while t[1] < tspan[1]
            #     popfirst!(t)
            # end
            # while t[end] > tspan[end]
            #     pop!(t)
            # end
            u = c.solution.states
            c.solution.success = (size(u) == (length(nfmu.x0), length(ts)))

            if size(u)[2] > 0 # at least some recorded points
                c.solution.states =
                    build_solution(prob, solver, ts, collect(u[:, i] for i = 1:size(u)[2]))
            end
        else
            c.solution.success = (c.solution.states.retcode == ReturnCode.Success)
        end

    end # ignore_derivatives

    @debug "ME_NeuralFMU: Stopping callback..."

    # stopCB 
    stopCallback(nfmu, c, t_stop)

    # cleanup snapshots to release memory
    if cleanSnapshots
        logInfo(c.fmu, "Lazy unloading $(length(c.solution.snapshots)) snapshots ...")

        for snapshot in c.solution.snapshots
            FMIBase.freeSnapshot!(snapshot)
        end
        # freeSnapshot only removes them from the FMU2Instance, not the FMUSolution
        c.solution.snapshots = Vector{FMUSnapshot}(undef, 0)
    end

    # if nfmu.fmu.isDummyDiscrete
    #     logInfo(nfmu.fmu, "De-activating dummy discrete state for FMU to perform proper simulation.")
    #     nfmu.fmu.isDummyDiscrete = false
    # end

    return c.solution
end
function (nfmu::ME_NeuralFMU)(x0::Union{Array{<:Real},Nothing}, t::Real; p = nothing)

    c = nothing

    return fx(nfmu, c, x0, p, t)
end

"""

    ToDo: Docstring for Arguments, Keyword arguments, ...

Evaluates the CS_NeuralFMU in the timespan given during construction or in a custum timespan from `t_start` to `t_stop` with a given time step size `t_step`.

Via optional argument `reset`, the FMU is reset every time evaluation is started (default=`true`).
"""
function (nfmu::CS_NeuralFMU{F,C})(
    inputFct,
    t_step::Real,
    tspan::Tuple{Float64,Float64} = nfmu.tspan;
    p = nfmu.p,
    tolerance::Union{Real,Nothing} = nothing,
    parameters::Union{Dict{<:Any,<:Any},Nothing} = nothing,
) where {F,C}

    t_start, t_stop = tspan

    c = (hasCurrentInstance(nfmu.fmu) ? getCurrentInstance(nfmu.fmu) : nothing)
    c, _ = prepareSolveFMU(
        nfmu.fmu,
        c,
        fmi2TypeCoSimulation;
        parameters = parameters,
        t_start = t_start,
        t_stop = t_stop,
        tolerance = tolerance,
        cleanup = true,
    )

    ts = collect(t_start:t_step:t_stop)

    model_input = inputFct.(ts)

    firstStep = true
    function simStep(input)
        y = nothing

        if !firstStep
            ignore_derivatives() do
                fmi2DoStep(c, t_step)
            end
        else
            firstStep = false
        end

        if p == nothing # structured, implicite parameters
            y = nfmu.model(input)
        else # flattened, explicite parameters
            @assert !isnothing(nfmu.re) "Using explicite parameters without destructing the model."
            y = FMIFlux.evaluate(nfmu, input; p = p)
        end

        return y
    end

    valueStack = simStep.(model_input)

    ignore_derivatives() do
        c.solution.success = true
    end

    c.solution.values = SavedValues{typeof(ts[1]),typeof(valueStack[1])}(ts, valueStack)

    # [ToDo] check if this is still the case for current releases of related libraries
    # this is not possible in CS (pullbacks are sometimes called after the finished simulation), clean-up happens at the next call
    # c = finishSolveFMU(nfmu.fmu, c, freeInstance, terminate)

    return c.solution
end

function (nfmu::CS_NeuralFMU{Vector{F},Vector{C}})(
    inputFct,
    t_step::Real,
    tspan::Tuple{Float64,Float64} = nfmu.tspan;
    p = nothing,
    tolerance::Union{Real,Nothing} = nothing,
    parameters::Union{Vector{Union{Dict{<:Any,<:Any},Nothing}},Nothing} = nothing,
) where {F,C}

    t_start, t_stop = tspan
    numFMU = length(nfmu.fmu)

    cs = nothing
    ignore_derivatives() do
        cs = Vector{Union{FMU2Component,Nothing}}(undef, numFMU)
        for i = 1:numFMU
            cs[i] = (
                hasCurrentInstance(nfmu.fmu[i]) ? getCurrentInstance(nfmu.fmu[i]) : nothing
            )
        end
    end
    for i = 1:numFMU
        cs[i], _ = prepareSolveFMU(
            nfmu.fmu[i],
            cs[i],
            fmi2TypeCoSimulation;
            parameters = parameters,
            t_start = t_start,
            t_stop = t_stop,
            tolerance = tolerance,
            cleanup = true,
        )
    end

    solution = FMUSolution(nothing)

    ts = collect(t_start:t_step:t_stop)
    model_input = inputFct.(ts)

    firstStep = true
    function simStep(input)
        y = nothing

        if !firstStep
            ignore_derivatives() do
                for c in cs
                    fmi2DoStep(c, t_step)
                end
            end
        else
            firstStep = false
        end

        if p == nothing # structured, implicite parameters
            y = nfmu.model(input)
        else # flattened, explicite parameters
            @assert nfmu.re != nothing "Using explicite parameters without destructing the model."

            # if length(p) == 1
            #     y = nfmu.re(p[1])(input)
            # else
            #     y = nfmu.re(p)(input)
            # end
            FMIFlux.evaluate(nfmu, input; p = p)
        end

        return y
    end

    valueStack = simStep.(model_input)

    ignore_derivatives() do
        solution.success = true # ToDo: Check successful simulation!
    end

    solution.values = SavedValues{typeof(ts[1]),typeof(valueStack[1])}(ts, valueStack)

    # [ToDo] check if this is still the case for current releases of related libraries
    # this is not possible in CS (pullbacks are sometimes called after the finished simulation), clean-up happens at the next call
    # cs = finishSolveFMU(nfmu.fmu, cs, freeInstance, terminate)

    return solution
end

function gradientValidationFailure(
    jac,
    multiObjective::Bool,
    grad_threshold::Real,
    gradient::Symbol,
)
    grads = if multiObjective
        collect(jac[i, :] for i = 1:size(jac)[1])
    else
        [jac]
    end

    has_nothing =
        any(collect(any(isnothing.(grad)) for grad in grads)) || any(isnothing.(grads))
    if has_nothing
        return "Gradient determination with $(gradient) failed, because gradient contains `nothing`.\nNo back-up options available."
    end

    all_zero = any(collect(all(iszero.(grad)) for grad in grads))
    if all_zero
        return "Determined gradient containes only zeros.\nThis might be because the loss function is:\n(a) not sensitive regarding the model parameters or\n(b) sensitivities regarding the model parameters are not traceable via AD."
    end

    has_nan = any(collect(any(isnan.(grad)) for grad in grads))
    if has_nan
        return "Gradient determination with $(gradient) failed, because gradient contains `NaNs`.\nNo back-up options available."
    end

    max_grad = max(collect(max(abs.(grad)...) for grad in grads)...)
    if max_grad > grad_threshold
        return "Determined gradient max entry $(max_grad) > $(grad_threshold).\nGradient is rated as invalid, asserting."
    end

    return nothing
end

function computeGradient!(
    jac,
    loss,
    params,
    gradient::Symbol,
    chunk_size::Union{Symbol,Int},
    multiObjective::Bool,
    grad_threshold::Real,
)

    @assert !issense(params) "Called `computeGradient!` with AD-sensitive training parameters, resulting in nested AD. This is not supported for now."

    if gradient == :ForwardDiff

        if chunk_size == :auto_forwarddiff

            if multiObjective
                conf = ForwardDiff.JacobianConfig(loss, params)
                ForwardDiff.jacobian!(jac, loss, params, conf)
            else
                conf = ForwardDiff.GradientConfig(loss, params)
                ForwardDiff.gradient!(jac, loss, params, conf)
            end

        elseif chunk_size == :auto_fmiflux

            chunk_size = DEFAULT_CHUNK_SIZE

            if multiObjective
                conf = ForwardDiff.JacobianConfig(
                    loss,
                    params,
                    ForwardDiff.Chunk{min(chunk_size, length(params))}(),
                )
                ForwardDiff.jacobian!(jac, loss, params, conf)
            else
                conf = ForwardDiff.GradientConfig(
                    loss,
                    params,
                    ForwardDiff.Chunk{min(chunk_size, length(params))}(),
                )
                ForwardDiff.gradient!(jac, loss, params, conf)
            end
        else

            if multiObjective
                conf = ForwardDiff.JacobianConfig(
                    loss,
                    params,
                    ForwardDiff.Chunk{min(chunk_size, length(params))}(),
                )
                ForwardDiff.jacobian!(jac, loss, params, conf)
            else
                conf = ForwardDiff.GradientConfig(
                    loss,
                    params,
                    ForwardDiff.Chunk{min(chunk_size, length(params))}(),
                )
                ForwardDiff.gradient!(jac, loss, params, conf)
            end
        end

    elseif gradient == :Zygote

        if multiObjective
            jac[:] = Zygote.jacobian(loss, params)[1]
        else
            jac[:] = Zygote.gradient(loss, params)[1]
        end

    elseif gradient == :ReverseDiff

        if multiObjective
            ReverseDiff.jacobian!(jac, loss, params)
        else
            ReverseDiff.gradient!(jac, loss, params)
        end

        failure = gradientValidationFailure(jac, multiObjective, grad_threshold, gradient)
        if !isnothing(failure) && !forceScalarEventCallbacks()
            # ReverseDiffAdjoint occasionally returns a huge callback/root
            # adjoint for vector event callbacks after an otherwise valid
            # optimizer update. Retrying only the bad gradient with scalar
            # callbacks keeps the normal fast path intact and preserves
            # ReverseDiff for both the ODE and FMU VJPs.
            withScalarEventCallbacks() do
                if multiObjective
                    ReverseDiff.jacobian!(jac, loss, params)
                else
                    ReverseDiff.gradient!(jac, loss, params)
                end
            end
        end
    elseif gradient == :FiniteDiff

        if multiObjective
            FiniteDiff.finite_difference_jacobian!(jac, loss, params)
        else
            FiniteDiff.finite_difference_gradient!(jac, loss, params)
        end
    else
        @assert false "Unknown `gradient=$(gradient)`, supported are `:ForwardDiff`, `:Zygote`, `:FiniteDiff` and `:ReverseDiff`."
    end

    ### check gradient is valid 

    # has_nan = any(collect(any(isnan.(grad)) for grad in grads))
    # has_nothing =
    #     any(collect(any(isnothing.(grad)) for grad in grads)) || any(isnothing.(grads))

    #@info "$(grads[1][1:10])"

    # if gradient != :ForwardDiff && (has_nan || has_nothing)

    #     @assert !has_nan     "Gradient determination with $(gradient) failed, because gradient contains `NaNs`.\nPlease try a smaller value for `FMIFlux.DUMMY_DT` (currently is $(FMIFlux.DUMMY_DT)), that roughly matches the time step of your system."
    #     @assert !has_nothing "Gradient determination with $(gradient) failed, because gradient contains `nothing`.\nPlease open an issue."

    # @warn "Gradient determination with $(gradient) failed, because gradient contains `NaNs` and/or `nothing`.\nThis might be because the FMU is throwing redundant events, which is currently not supported.\nTrying ForwardDiff as back-up.\nIf this message gets printed (almost) every step, consider using keyword `gradient=:ForwardDiff` to fix ForwardDiff as sensitivity system."
    # gradient = :ForwardDiff
    # computeGradient!(jac, loss, params, gradient, chunk_size, multiObjective)

    # if multiObjective
    #     grads = collect(jac[i, :] for i = 1:size(jac)[1])
    # else
    #     grads = [jac]
    # end
    #end

    failure = gradientValidationFailure(jac, multiObjective, grad_threshold, gradient)
    @assert isnothing(failure) failure

    return nothing
end

#lk_TrainApply = ReentrantLock()
function trainStep(
    loss,
    params,
    gradient,
    chunk_size,
    optim::FMIFlux.AbstractOptimiser,
    printStep,
    proceed_on_assert,
    multiObjective;
    assert_length = 4096,
)

    #global lk_TrainApply

    try

        #for j = 1:length(params)

        step = FMIFlux.apply!(optim, params; printStep = printStep)

        #lock(lk_TrainApply) do

        #params[:] .-= step
        params[:] = params - step

        #end

        #end

    catch e
        if proceed_on_assert
            msg = "$(e)"
            if assert_length != 0
                if length(msg) > assert_length
                    msg = first(msg, assert_length) * "..."
                end
            end
            @error "Training asserted, but continuing: $(msg)"
        else
            throw(e)
        end
    end

    return nothing
end

"""

    train!(loss, neuralFMU::Union{ME_NeuralFMU, CS_NeuralFMU}, data, optim; gradient::Symbol=:ReverseDiff, kwargs...)

A function analogous to Flux.train! but with additional features and explicit parameters (faster).

# Arguments
- `loss` a loss function in the format `loss(p)`
- `neuralFMU` a object holding the neuralFMU with its parameters
- `data` the training data (or often an iterator)
- `optim` the optimizer used for training 

# Keywords 
- `gradient` a symbol determining the AD-library for gradient computation, available are `:ForwardDiff`, `:Zygote` and :ReverseDiff (default)
- `cb` a custom callback function that is called after every training step (default `nothing`)
- `chunk_size` the chunk size for AD using ForwardDiff (ignored for other AD-methods) (default `:auto_fmiflux`)
- `printStep` a boolean determining wheater the gradient min/max is printed after every step (for gradient debugging) (default `false`)
- `proceed_on_assert` a boolean that determins wheater to throw an ecxeption on error or proceed training and just print the error (default `false`)
- `multiThreading`: a boolean that determins if multiple gradients are generated in parallel (default `false`)
- `multiObjective`: set this if the loss function returns multiple values (multi objective optimization), currently gradients are fired to the optimizer one after another (default `false`)
"""
function train!(
    loss,
    neuralFMU::Union{ME_NeuralFMU,CS_NeuralFMU},
    data,
    optim;
    gradient::Symbol = :ReverseDiff,
    grad_threshold::Real = 1e6,
    kwargs...,
)
    params = FMIFlux.params(neuralFMU)

    return train!(
        loss,
        neuralFMU,
        params,
        data,
        optim;
        gradient = gradient,
        grad_threshold = grad_threshold,
        kwargs...,
    )
end

# Dispatch for FMIFlux.jl [FMIFlux.AbstractOptimiser]
function train!(
    loss,
    neuralFMU::NeuralFMU,
    params, #::Union{Flux.Params,Zygote.Params,AbstractVector{<:AbstractVector{<:Real}}},
    data,
    optim::FMIFlux.AbstractOptimiser;
    gradient::Symbol = :ReverseDiff,
    cb = nothing,
    chunk_size::Union{Integer,Symbol} = :auto_fmiflux,
    printStep::Bool = false,
    proceed_on_assert::Bool = false,
    multiThreading::Bool = false,
    multiObjective::Bool = false,
    assert_length = 4096,
    useSnapshots::Bool = true,
)

    if length(params) <= 0 #|| length(params[1]) <= 0
        @warn "train!(...): Empty parameter array, training on an empty parameter array doesn't make sense."
        return
    end

    if multiThreading && Threads.nthreads() == 1
        @warn "train!(...): Multi-threading is set via flag `multiThreading=true`, but this Julia process does not have multiple threads. This will not result in a speed-up. Please spawn Julia in multi-thread mode to speed-up training."
    end

    _trainStep = function (i)
        # [Note] :ReverseDiff, :Zygote need it for state change sampling and the rrule
        #        :ForwardDiff needs it for state change sampling
        snapshots = neuralFMU.snapshots
        neuralFMU.snapshots = useSnapshots

        ret = trainStep(
            loss,
            params,
            gradient,
            chunk_size,
            optim,
            printStep,
            proceed_on_assert,
            multiObjective;
            assert_length = assert_length,
        )

        neuralFMU.snapshots = snapshots

        if !snapshots
            # invalidate all active snapshots, otherwise execConfig = no-reset causes massive memory allocations!
            if hasCurrentInstance(neuralFMU.fmu)
                c = getCurrentInstance(neuralFMU.fmu)

                logInfo(
                    c.fmu,
                    "Lazy unloading $(length(c.solution.snapshots)) snapshots ...",
                )

                # lazy unloading!
                # but only the solution snapshots, batch snapshots should be kept available
                for snapshot in c.solution.snapshots
                    freeSnapshot!(snapshot)
                end

                # freeSnapshot only removes them from the FMU2Instance, not the FMUSolution
                c.solution.snapshots = Vector{FMUSnapshot}(undef, 0)

            end

        end

        # clean things up (e.g. if asserting gets thrown during AD pass)
        neuralFMU.p = unsense(neuralFMU.p)

        if neuralFMU.fmu.isDummyDiscrete
            logInfo(
                neuralFMU.fmu,
                "De-activating dummy discrete state for FMU to perform proper simulation.",
            )
            neuralFMU.fmu.isDummyDiscrete = false
        end

        # call callbacks
        if cb != nothing
            if isa(cb, AbstractArray)
                for _cb in cb
                    _cb()
                end
            else
                cb()
            end
        end

        return ret
    end

    if multiThreading
        ThreadPools.qforeach(_trainStep, 1:length(data))
    else
        foreach(_trainStep, 1:length(data))
    end

end
