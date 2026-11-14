param(
    [switch]$SetupOnly,
    [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'
$browserJob = $null
Push-Location -LiteralPath $PSScriptRoot
try {
    if (-not (Test-Path -LiteralPath 'config.local.json')) {
        Copy-Item -LiteralPath 'config.example.json' -Destination 'config.local.json'
        Write-Host 'Created config.local.json. Fill the credentials locally before starting a cloud test.'
    }

    $probePython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $probePython)) {
        $projectPython = Join-Path $PSScriptRoot '..\..\.venv\Scripts\python.exe'
        if (Test-Path -LiteralPath $projectPython) {
            & $projectPython -m venv .venv
        } else {
            & python -m venv .venv
        }
        if ($LASTEXITCODE -ne 0) { throw 'Failed to create the isolated Python environment.' }
    }

    $requirementsHash = (Get-FileHash -LiteralPath 'requirements.txt' -Algorithm SHA256).Hash
    $installedHash = if (Test-Path -LiteralPath '.python-deps.sha256') { (Get-Content -LiteralPath '.python-deps.sha256' -Raw).Trim() } else { '' }
    if ($requirementsHash -ne $installedHash) {
        & $probePython -m pip install --disable-pip-version-check -r requirements.txt
        if ($LASTEXITCODE -ne 0) { throw 'Python dependency installation failed. Run this launcher again after fixing the network.' }
        Set-Content -LiteralPath '.python-deps.sha256' -Value $requirementsHash -Encoding ascii
    }

    $npmCommand = (Get-Command npm.cmd -ErrorAction Stop).Source
    $packageHash = (Get-FileHash -LiteralPath 'package.json' -Algorithm SHA256).Hash
    $nodeHash = if (Test-Path -LiteralPath '.node-deps.sha256') { (Get-Content -LiteralPath '.node-deps.sha256' -Raw).Trim() } else { '' }
    if (($packageHash -ne $nodeHash) -or (-not (Test-Path -LiteralPath 'node_modules'))) {
        if (Test-Path -LiteralPath 'package-lock.json') {
            & $npmCommand ci --no-audit --no-fund
        } else {
            & $npmCommand install --no-audit --no-fund
        }
        if ($LASTEXITCODE -ne 0) { throw 'Node dependency installation failed. Run this launcher again after fixing the network.' }
        Set-Content -LiteralPath '.node-deps.sha256' -Value $packageHash -Encoding ascii
    }
    & $npmCommand run build
    if ($LASTEXITCODE -ne 0) { throw 'The validation page build failed.' }

    if ($SetupOnly) {
        Write-Host 'Local setup and build finished. No cloud session was opened.'
        exit 0
    }

    Write-Host 'Open http://localhost:18765 in Chrome or Edge. Press Ctrl+C here to stop the local server.'
    if (-not $NoBrowser) {
        $browserJob = Start-Job -ScriptBlock {
            for ($attempt = 0; $attempt -lt 40; $attempt++) {
                try {
                    $health = Invoke-RestMethod -Uri 'http://localhost:18765/api/health' -TimeoutSec 1
                    if ($health) { Start-Process 'http://localhost:18765'; return }
                } catch { }
                Start-Sleep -Milliseconds 250
            }
        }
    }
    & $probePython -m uvicorn server:create_app --factory --host 127.0.0.1 --port 18765 --no-access-log
    if ($LASTEXITCODE -ne 0) { throw 'Local server stopped with an error. Check whether port 18765 is already in use.' }
} finally {
    if ($null -ne $browserJob) {
        Stop-Job -Job $browserJob -ErrorAction SilentlyContinue
        Remove-Job -Job $browserJob -Force -ErrorAction SilentlyContinue
    }
    Pop-Location
}
