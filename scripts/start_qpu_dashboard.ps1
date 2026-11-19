# Start the Q-FraudGuard dashboard on a QPU backend after a short deployment check.
#   scripts\start_qpu_dashboard.ps1 -Solver quafu-qpu      (Quafu-SQC; needs QPU_API_TOKEN)
#   scripts\start_qpu_dashboard.ps1 -Solver dwave-qpu      (D-Wave Leap; needs DWAVE_API_TOKEN)
# Tokens are read from the environment or the git-ignored .env and never printed.
param(
    [ValidateSet("quafu-qpu", "quafu-mock", "quafu-anneal-qpu", "quafu-anneal-mock", "quafu-fast-qpu", "quafu-fast-mock", "dwave-qpu", "dwave-hybrid", "dwave-mock")]
    [string]$Solver = "quafu-qpu",
    [string]$Dataset = "amlworld_full",
    [int]$CheckWindows = 2,
    [ValidateRange(1, 2147483647)]
    [int]$WindowSize
)

$root = Split-Path -Parent $PSScriptRoot
$env:QFG_SOLVER = $Solver
$env:QFG_PRIORITY_MODE = "qubo"
$env:QFG_DATASET = $Dataset
if ($Solver -like "quafu*") {
    # the Quafu SDK (quarkstudio) is installed only in the isolated environment
    $python = Join-Path $root ".venv-quafu\Scripts\python.exe"
} else {
    $python = "python"
}

Set-Location $root
$checkArgs = @("--solver", $Solver, "--windows", $CheckWindows)
if ($PSBoundParameters.ContainsKey("WindowSize")) {
    $env:QFG_QUANTUM_MIN_BATCH = [string]$WindowSize
    $env:QFG_QUANTUM_MAX_BATCH = [string]$WindowSize
    $checkArgs += @("--window-size", $WindowSize)
}
& $python scripts\qpu_check.py @checkArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python -m streamlit run dashboard\app.py
