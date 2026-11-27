#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using FMIFlux
using Test
using FMIZoo
using FMIFlux.FMIImport
using FMIFlux.FMIImport.FMICore
import Flux

import FMIFlux.FMISensitivity: FiniteDiff, ForwardDiff, ReverseDiff

using FMIFlux.FMIImport:
    stringToValueReference, fmi2ValueReference, prepareSolveFMU, fmi2Real
using FMIFlux.FMIImport: FMU_EXECUTION_CONFIGURATIONS
using FMIFlux.FMIImport: getState, getValue, getTime

exportingToolsWindows = [("Dymola", "2022x")] # [("ModelicaReferenceFMUs", "0.0.25")]
exportingToolsLinux = [("Dymola", "2022x")]

# number of training steps to perform
global NUMSTEPS = 30
global ETA = 1e-4 # 1e-5
global GRADIENT = nothing
global EXPORTINGTOOL = nothing
global EXPORTINGVERSION = nothing
global X0 = [2.0, 0.0]
global OPTIMISER = Flux.Descent
# Gradient based optimisation does not guarantee a smaller loss after every step (e.g. overshooting
# near events), so this is only a smoke test for training. The gradients themselves are checked
# with `gradientCheckRatio` before training.
global FAILED_GRADIENTS_QUOTA = 1 / 3

# callback for bad optimization steps counter
global FAILED_GRADIENTS = 0
global LAST_LOSS
callback = function (p)
    global LAST_LOSS, FAILED_GRADIENTS

    #@info "Start computing loss ..."
    loss = losssum(p) # p[1]
    #@info "Finished computing loss = $(loss)"

    if loss == LAST_LOSS
        @error "Loss stays unchanged.\nZero gradient or broken Optimiser implementation."
        FAILED_GRADIENTS += 1
    elseif loss > LAST_LOSS
        FAILED_GRADIENTS += 1
    end
    LAST_LOSS = loss
    return false
end

# Gradient check, see `gradientCheckRatio`.
import LinearAlgebra
# relative change of the loss for the finite-difference probes along the gradient direction
# (measured at the start configurations of hybrid_ME.jl / hybrid_ME_dis.jl: ratios within
# [0.98, 1.006] for δ = 1e-5, larger steps may cross events or become nonlinear)
global GRADIENT_CHECK_DELTA = 1e-5
# accepted deviation of the ratio from 1
global GRADIENT_CHECK_ATOL = 0.1

# An optimiser that only records the gradient of a training step and does not change the parameters.
mutable struct GradientRecorder <: FMIFlux.AbstractOptimiser
    loss::Any
    gradient::Symbol
    grad::Vector{Float64}
end
function FMIFlux.apply!(o::GradientRecorder, params; printStep::Bool = false)
    FMIFlux.computeGradient!(o.grad, o.loss, params, o.gradient, :auto_fmiflux, false, 1e6)
    return zero(params)
end

"""
Checks the training gradient `g` of `loss` at `p` against the loss itself: the central finite
difference of the loss along the gradient direction must equal `|g|`, so the returned ratio is ≈ 1.
Unlike counting training steps with increasing loss, this checks the gradient (direction and
magnitude) and not the optimiser. It costs one gradient and three loss evaluations instead of
2n for a finite-difference gradient. The step is chosen such that the loss changes by about
`δ⋅loss(p)`, which keeps it well above the solver noise while not crossing events.
"""
function gradientCheckRatio(
    loss,
    neuralFMU,
    p;
    gradient::Symbol = GRADIENT,
    δ::Real = GRADIENT_CHECK_DELTA,
)
    L0 = loss(p) # every solve stores its parameters in the NeuralFMU (`nfmu.p = p`), so this also selects `p` for training

    # one training step that does not change the parameters: the gradient is determined
    # exactly as during training (e.g. with snapshots enabled)
    recorder = GradientRecorder(loss, gradient, zeros(length(p)))
    FMIFlux.train!(
        loss,
        neuralFMU,
        p,
        Iterators.repeated((), 1),
        recorder;
        gradient = gradient,
    )
    g = recorder.grad

    ng = LinearAlgebra.norm(g)
    if !isfinite(ng) || ng == 0.0
        loss(p)
        return ng == 0.0 ? 0.0 : NaN
    end

    d = g ./ ng
    h = δ * abs(L0) / ng
    ratio = (loss(p .+ h .* d) - loss(p .- h .* d)) / (2 * h * ng)

    loss(p) # restore `p` in the NeuralFMU
    return ratio
end

# training data for pendulum experiment
function syntTrainingData(tData)
    posData = cos.(tData * 3.0) * 2.0
    velData = sin.(tData * 3.0) * -6.0
    accData = cos.(tData * 3.0) * -18.0
    return posData, velData, accData
end

# enable assertions for warnings/errors for all default execution configurations - we want strict tests here
for exec in FMU_EXECUTION_CONFIGURATIONS
    exec.assertOnError = true
    exec.assertOnWarning = true
end

# dummy function to test the implementation of the isolated event functions,
# the integrator is simplified with a NamedTuple here.
import FMIFlux: u_modified!
function FMIFlux.u_modified!(integrator::NamedTuple, state::Bool)
    nothing
end

function runtests(exportingTool)

    global EXPORTINGTOOL = exportingTool[1]
    global EXPORTINGVERSION = exportingTool[2]
    @info "Testing FMUs exported from $(EXPORTINGTOOL) ($(EXPORTINGVERSION))"
    @testset "Testing FMUs exported from $(EXPORTINGTOOL) ($(EXPORTINGVERSION))" begin

        @info "Solution Gradients (solution_gradients.jl)"
        @testset "Solution Gradients" begin
            include("solution_gradients.jl")
        end

        @info "Time Event Solution Gradients (time_solution_gradients.jl)"
        @testset "Time Event Solution Gradients" begin
            include("time_solution_gradients.jl")
        end

        @info "Layers (layers.jl)"
        @testset "Layers" begin
            include("layers.jl")
        end

        @info "Loading / Saving (load_save.jl)"
        @testset "Loading / Saving" begin
            include("load_save.jl")
        end

        @info "Snapshots (snapshots.jl)"
        @testset "Snapshots" begin
            include("snapshots.jl")
        end

        for _GRADIENT ∈ (:ReverseDiff, :ForwardDiff) # (:ReverseDiff, ) # 

            global GRADIENT = _GRADIENT
            @info "Gradient: $(GRADIENT)"
            @testset "Gradient: $(GRADIENT)" begin

                @info "ME-NeuralFMU (Continuous) (hybrid_ME.jl)"
                @testset "ME-NeuralFMU (Continuous)" begin
                    include("hybrid_ME.jl")
                end

                @info "ME-NeuralFMU (Discontinuous) (hybrid_ME_dis.jl)"
                @testset "ME-NeuralFMU (Discontinuous)" begin
                    include("hybrid_ME_dis.jl")
                end

                @info "NeuralFMU with FMU parameter optimization (fmu_params.jl)"
                @testset "NeuralFMU with FMU parameter optimization" begin
                    include("fmu_params.jl")
                end

                @info "Training modes (train_modes.jl)"
                @testset "Training modes" begin
                    include("train_modes.jl")
                end

                @info "CS-NeuralFMU (hybrid_CS.jl)"
                @testset "CS-NeuralFMU" begin
                    include("hybrid_CS.jl")
                end

                @info "Batching (batching.jl)"
                @testset "Batching" begin
                    include("batching.jl")
                end

                @warn "Multi-threading test skipped"
                # @info    "Multi-threading (multi_threading.jl)"
                # @testset "Multi-threading" begin
                #     include("multi_threading.jl")
                # end

                @warn "Multiple FMUs test skipped"
                # @info "Multiple FMUs (multi.jl)"
                # @testset "Multiple FMUs" begin
                #     include("multi.jl")
                # end

                @warn "Optim.jl tests disabled, need fixing."
                # @info "Optimizers from Optim.jl (optim.jl)"
                # @testset "Optim" begin
                #     include("optim.jl")
                # end

            end
        end

        @info "Checking supported sensitivities skipped"
        # @info    "Benchmark: Supported sensitivities (supported_sensitivities.jl)"
        # @testset "Benchmark: Supported sensitivities " begin
        #     include("supported_sensitivities.jl")
        # end

    end
end

@testset "FMIFlux.jl" begin
    if Sys.iswindows()
        @info "Automated testing is supported on Windows."
        for exportingTool in exportingToolsWindows
            runtests(exportingTool)
        end
    elseif Sys.islinux()
        @info "Automated testing is supported on Linux."
        for exportingTool in exportingToolsLinux
            runtests(exportingTool)
        end
    elseif Sys.isapple()
        @warn "Test-sets are currrently using Windows- and Linux-FMUs, automated testing for macOS is currently not supported."
    end
end
