# Start the Q-FraudGuard dashboard locally on CPU. Select a provider explicitly for a hardware run.
#   scripts\start_qpu_dashboard.ps1 -Solver quafu-qpu      (Quafu-SQC; needs QPU_API_TOKEN)
#   scripts\start_qpu_dashboard.ps1 -Solver dwave-qpu      (D-Wave Leap; needs DWAVE_API_TOKEN)
# Tokens are read from the environment or the git-ignored .env and never printed.
param(
    [ValidateSet("exact", "sqa", "quafu-qpu", "quafu-mock", "quafu-anneal-qpu", "quafu-anneal-mock", "quafu-fast-qpu", "quafu-fast-mock", "dwave-qpu", "dwave-hybrid", "dwave-mock")]
    [string]$Solver = "exact",
    [string]$Dataset = "amlworld_full",
    [int]$CheckWindows = 2,
    [ValidateRange(1, 2147483647)]
    [int]$WindowSize
)

$root = Split-Path -Parent $PSScriptRoot
$env:QFG_SOLVER = $Solver
$env:QFG_PRIORITY_MODE = "qubo"
$env:QFG_DATASET = $Dataset
if ($env:QFG_SOURCE -eq "recorded" -and $Solver -notin @("exact", "sqa")) {
    $env:QFG_SOURCE = "replay"
}
$quafuPython = Join-Path $root ".venv-quafu\Scripts\python.exe"
if ($Solver -like "quafu*" -or ($Solver -eq "exact" -and (Test-Path -LiteralPath $quafuPython))) {
    # the Quafu SDK (quarkstudio) is installed only in the isolated environment
    $python = $quafuPython
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
if ($Solver -notin @("exact", "sqa")) {
    & $python scripts\qpu_check.py @checkArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
& $python -m streamlit run dashboard\app.py
