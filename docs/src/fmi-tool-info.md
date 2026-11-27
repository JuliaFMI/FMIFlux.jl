# FMI Compatibility Information

!!! note
    This page is provided and maintained by the developers of *FMIFlux.jl* ([JuliaFMI](https://github.com/JuliaFMI)). It is not provided by the Modelica Association or the FMI Project.

*FMIFlux.jl* is an FMU importer: it combines FMUs with machine learning models (NeuralFMUs) and does not export FMUs.
FMUs are loaded and simulated with [*FMIImport.jl*](https://github.com/JuliaFMI/FMIImport.jl), the import backend of [*FMI.jl*](https://github.com/JuliaFMI/FMI.jl).
Therefore, the [compatibility information of *FMI.jl*](https://juliafmi.github.io/FMI.jl/stable/fmi-tool-info/) (FMI Cross-Check results and the *FMIImport.jl* test suite for FMI 2.0 and FMI 3.0) also applies to *FMIFlux.jl*.

## FMU Import

The [test suite of *FMIFlux.jl*](https://github.com/JuliaFMI/FMIFlux.jl/tree/main/test) additionally checks the functionality specific to *FMIFlux.jl* (training of NeuralFMUs, sensitivities through FMUs via [*FMISensitivity.jl*](https://github.com/JuliaFMI/FMISensitivity.jl)) with the following FMUs.
The tests run automatically in GitHub Actions on Windows (`win64`) and Linux (`linux64`), with the latest Julia release and the Julia LTS release.

| Exporting tool | Tool version | FMI version | Interface type | FMUs | Tested functionality |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Dymola ([FMUs](https://github.com/JuliaFMI/FMIZoo.jl/tree/main/models/bin/Dymola/2022x)) | 2022x | 2.0 | ME | `SpringPendulum1D`, `SpringFrictionPendulum1D` | NeuralFMU training, FMU parameter optimization, batching, training modes, loading/saving of NeuralFMUs |
| Dymola ([FMUs](https://github.com/JuliaFMI/FMIZoo.jl/tree/main/models/bin/Dymola/2022x)) | 2022x | 2.0 | CS | `SpringPendulumExtForce1D` | NeuralFMU training |
| Dymola ([FMUs](https://github.com/JuliaFMI/FMIZoo.jl/tree/main/models/bin/Dymola/2023x)) | 2023x | 2.0 | ME | `BouncingBall1D`, `BouncingBallGravitySwitch1D` | NeuralFMUs with state and time events, solution gradients through events, FMU snapshots |

The test suite of *FMIFlux.jl* currently uses FMI 2.0 FMUs. Importing and simulating FMI 3.0 FMUs is covered by the *FMIImport.jl* test suite (see the [compatibility information of *FMI.jl*](https://juliafmi.github.io/FMI.jl/stable/fmi-tool-info/)).

### Examples

The [examples](examples/overview.md) of this documentation are executed by the [examples workflow](https://github.com/JuliaFMI/FMIFlux.jl/actions/workflows/Example.yml) (Windows, Julia 1.10) with FMI 2.0 Model Exchange and Co-Simulation FMUs exported by Dymola (provided by [*FMIZoo.jl*](https://github.com/JuliaFMI/FMIZoo.jl/tree/main/models/bin/Dymola)):

| Exporting tool | Tool version | FMI version | FMUs |
| :--- | :--- | :--- | :--- |
| Dymola | 2020x | 2.0 | `VLDM` |
| Dymola | 2022x | 2.0 | `SpringFrictionPendulum1D`, `SpringPendulum1D`, `SpringPendulumExtForce1D` |
| Dymola | 2023x | 2.0 | `RobotRR` |

## Reproducing the Results

Run `using Pkg; Pkg.test("FMIFlux")` in the Julia REPL. The FMUs are downloaded automatically via [*FMIZoo.jl*](https://github.com/JuliaFMI/FMIZoo.jl).
