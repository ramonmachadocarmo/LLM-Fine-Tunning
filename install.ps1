# Check and install host dependencies for LLM Fine-Tuning Engine (Windows PowerShell).
# Usage: .\install.ps1 [-Setup]
# ASCII-only strings (Windows PowerShell 5.1 + UTF-8 without BOM).

[CmdletBinding()]
param(
    [switch]$Setup,
    [switch]$Help
)

$ErrorActionPreference = "Stop"
$PyVer = if ($env:PY_VER) { $env:PY_VER } else { "3.11.9" }
$Root = $PSScriptRoot
Set-Location $Root

if ($Help) {
    Write-Host "Usage: .\install.ps1 [-Setup]"
    Write-Host "  Checks make, curl, git, pyenv-win, Poetry, Python $PyVer."
    Write-Host "  -Setup  also runs: make setup"
    exit 0
}

function Write-Ok($msg)   { Write-Host "  [OK]  $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "  [!!]  $msg" -ForegroundColor Yellow }
function Write-Fail($msg) { Write-Host "  [XX]  $msg" -ForegroundColor Red }

function Test-Cmd($name) {
    return [bool](Get-Command $name -ErrorAction SilentlyContinue)
}

Write-Host "==> LLM Fine-Tuning Engine - dependency check"
$missing = 0

if (Test-Cmd "make") { Write-Ok "make" } else { Write-Fail "make not found (install Git for Windows or chocolatey: choco install make)"; $missing++ }
if (Test-Cmd "curl") { Write-Ok "curl" } else { Write-Fail "curl not found"; $missing++ }
if (Test-Cmd "git")  { Write-Ok "git $((git --version 2>$null))"; } else { Write-Fail "git not found"; $missing++ }

# --- pyenv-win ---
$pyenvRoot = Join-Path $env:USERPROFILE ".pyenv\pyenv-win"
$env:PYENV = $pyenvRoot
$env:PYENV_ROOT = $pyenvRoot
$env:PYENV_HOME = $pyenvRoot
$env:PATH = @(
    (Join-Path $pyenvRoot "bin"),
    (Join-Path $pyenvRoot "shims"),
    $env:PATH
) -join ";"

if (-not (Test-Cmd "pyenv")) {
    Write-Warn "pyenv-win missing"
    $installer = Join-Path $env:TEMP "install-pyenv-win.ps1"
    try {
        Write-Warn "downloading pyenv-win installer..."
        Invoke-WebRequest -UseBasicParsing -Uri "https://raw.githubusercontent.com/pyenv-win/pyenv-win/master/pyenv-win/install-pyenv-win.ps1" -OutFile $installer
        & powershell -NoProfile -ExecutionPolicy Bypass -File $installer
        $env:PATH = @(
            (Join-Path $pyenvRoot "bin"),
            (Join-Path $pyenvRoot "shims"),
            $env:PATH
        ) -join ";"
    } catch {
        Write-Fail "pyenv-win install failed: $_"
        Write-Fail "Manual: https://github.com/pyenv-win/pyenv-win"
        $missing++
    }
}

if (Test-Cmd "pyenv") {
    Write-Ok "pyenv $((pyenv --version 2>$null))"
    $pyHome = Join-Path $pyenvRoot "versions\$PyVer"
    if (-not (Test-Path (Join-Path $pyHome "python.exe"))) {
        Write-Warn "installing Python $PyVer via pyenv (may take a few minutes)"
        pyenv install -s $PyVer
    } else {
        Write-Ok "Python $PyVer already installed via pyenv"
    }
    if (Test-Path (Join-Path $pyHome "python.exe")) {
        pyenv local $PyVer 2>$null
        $env:PATH = @(
            (Join-Path $pyenvRoot "bin"),
            (Join-Path $pyenvRoot "shims"),
            $pyHome,
            (Join-Path $pyHome "Scripts"),
            $env:PATH
        ) -join ";"
        Write-Ok "pyenv local -> $PyVer"
    } else {
        Write-Fail "Python $PyVer not available after pyenv install"
        $missing++
    }
} else {
    Write-Fail "pyenv still unavailable"
    $missing++
}

# --- Poetry ---
if (Test-Cmd "poetry") {
    Write-Ok "poetry $((poetry --version 2>$null))"
} else {
    Write-Warn "Poetry missing - installing"
    try {
        (Invoke-WebRequest -UseBasicParsing -Uri https://install.python-poetry.org).Content | python -
        $poetryBin = Join-Path $env:APPDATA "Python\Scripts"
        if (Test-Path $poetryBin) {
            $env:PATH = "$poetryBin;$env:PATH"
        }
        $poetryLocal = Join-Path $env:USERPROFILE ".local\bin"
        if (Test-Path $poetryLocal) {
            $env:PATH = "$poetryLocal;$env:PATH"
        }
    } catch {
        Write-Fail "Poetry install failed: $_"
        $missing++
    }
    if (Test-Cmd "poetry") {
        Write-Ok "poetry $((poetry --version 2>$null))"
    } else {
        Write-Fail "Poetry not on PATH after install (open a new shell or add %APPDATA%\Python\Scripts)"
        $missing++
    }
}

if ($missing -gt 0) {
    Write-Host ""
    Write-Host "Some required tools are missing. Fix them and re-run .\install.ps1"
    exit 1
}

Write-Host ""
Write-Host "==> host deps OK"
Write-Host "    Next: make setup && make up"
Write-Host "    Or:   .\install.ps1 -Setup"

if ($Setup) {
    Write-Host ""
    Write-Host "==> make setup"
    make setup
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
