param(
    [switch]$VerboseOutput
)

$ErrorActionPreference = "Stop"

function Write-Status {
    param(
        [string]$Level,
        [string]$Message
    )
    Write-Host "[$Level] $Message"
}

function Pass($msg) { Write-Status -Level "PASS" -Message $msg }
function Warn($msg) { Write-Status -Level "WARN" -Message $msg }
function Fail($msg) { Write-Status -Level "FAIL" -Message $msg }

$projectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $projectRoot

Write-Host "Project root: $projectRoot"

$failed = $false

# 1) Python availability
try {
    $pythonVersion = & python --version 2>&1
    Pass "Python detected: $pythonVersion"
} catch {
    Fail "Python is not available in PATH."
    exit 1
}

# 2) Required files
$requiredFiles = @(
    "app.py",
    "app_ai.py",
    "dashboard.py",
    "clinical_utils.py",
    "requirements.txt",
    ".env.example"
)

foreach ($file in $requiredFiles) {
    if (Test-Path $file) {
        Pass "Found file: $file"
    } else {
        Fail "Missing required file: $file"
        $failed = $true
    }
}

# 3) Environment file presence
if (Test-Path ".env") {
    Pass ".env file found."
} else {
    Warn ".env file not found. Create it from .env.example before running services."
}

# 4) Dependency import check
$imports = @(
    "flask",
    "openai",
    "twilio",
    "psycopg2",
    "streamlit",
    "pandas",
    "matplotlib",
    "sklearn"
)

$importFailures = @()
foreach ($pkg in $imports) {
    $cmd = "import $pkg"
    $result = & python -c $cmd 2>$null
    if ($LASTEXITCODE -ne 0) {
        $importFailures += $pkg
    }
}

if ($importFailures.Count -eq 0) {
    Pass "Core Python package imports succeeded."
} else {
    Warn ("Missing/unavailable imports: " + ($importFailures -join ", "))
}

# 5) Optional output artefacts
$expectedArtifacts = @(
    "test_dataset_mds04_native.csv",
    "mds04_native_test_report_results.csv",
    "mds04_native_confusion_matrix_evaluation.png"
)

foreach ($artifact in $expectedArtifacts) {
    if (Test-Path $artifact) {
        Pass "Artifact present: $artifact"
    } else {
        Warn "Artifact not found: $artifact"
    }
}

# 6) Port sanity for local runs
$portsToCheck = @(5000, 8501)
foreach ($port in $portsToCheck) {
    try {
        $used = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
        if ($used) {
            Warn "Port $port appears in use."
        } else {
            Pass "Port $port is currently free."
        }
    } catch {
        Warn "Could not inspect port $port on this system."
    }
}

if ($VerboseOutput) {
    Write-Host ""
    Write-Host "Suggested commands:"
    Write-Host "  pip install -r requirements.txt"
    Write-Host "  python app.py"
    Write-Host "  python -m streamlit run dashboard.py"
}

if ($failed) {
    Fail "Quick check completed with blocking failures."
    exit 2
}

Pass "Quick check completed."
exit 0
