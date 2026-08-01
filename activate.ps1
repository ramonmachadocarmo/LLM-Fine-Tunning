# Ativa pyenv 3.11.9 + Poetry neste projeto (Windows PowerShell)
# Uso: . .\activate.ps1

$ErrorActionPreference = "Stop"
$pyenvRoot = Join-Path $env:USERPROFILE ".pyenv\pyenv-win"
$pyVer = "3.11.9"
$pyHome = Join-Path $pyenvRoot "versions\$pyVer"

if (-not (Test-Path "$pyHome\python.exe")) {
    throw "Python $pyVer nao encontrado em $pyHome. Rode: pyenv install $pyVer"
}

# pyenv primeiro — evita Scoop / poetry quebrado do AppData
$env:PYENV = $pyenvRoot
$env:PYENV_ROOT = $pyenvRoot
$env:PYENV_HOME = $pyenvRoot
$env:PATH = @(
    (Join-Path $pyenvRoot "bin"),
    (Join-Path $pyenvRoot "shims"),
    $pyHome,
    (Join-Path $pyHome "Scripts"),
    ($env:PATH -split ";" | Where-Object {
        $_ -and
        $_ -notmatch 'scoop\\apps\\python' -and
        $_ -notmatch 'AppData\\Roaming\\Python\\Scripts'
    })
) -join ";"

Set-Location $PSScriptRoot
if (Get-Command pyenv -ErrorAction SilentlyContinue) {
    pyenv local $pyVer | Out-Null
}

Write-Host "Python : $(python --version)  ($(Get-Command python).Source)"
Write-Host "Poetry : $(poetry --version)  ($(Get-Command poetry).Source)"
if (Test-Path ".\.venv\Scripts\python.exe") {
    Write-Host "Venv   : .\.venv  ($((.\.venv\Scripts\python.exe --version) 2>&1))"
} else {
    Write-Host "Venv   : ausente — rode: make setup"
}
Write-Host ""
Write-Host "Comandos: make setup | make up | make train | make down"
Write-Host "(Linux/macOS: source ./activate.sh)"
