param(
    [switch]$Release
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Paper = Join-Path $Root "publications\paper"
$Build = Join-Path $Paper "build"
$Releases = Join-Path $Paper "releases"

New-Item -ItemType Directory -Force -Path $Build, $Releases | Out-Null

& python (Join-Path $PSScriptRoot "sync_experimental_results.py") --check
if ($LASTEXITCODE -ne 0) {
    throw "Experimental result outputs are stale. Run sync_experimental_results.py --write."
}

Push-Location $Paper
try {
    foreach ($pass in 1..3) {
        & pdflatex -disable-installer -interaction=nonstopmode -halt-on-error `
            -output-directory=build main.tex
        if ($LASTEXITCODE -ne 0) {
            throw "pdflatex pass $pass failed."
        }
    }
}
finally {
    Pop-Location
}

if ($Release) {
    Copy-Item -Force -LiteralPath (Join-Path $Build "main.pdf") `
        -Destination (Join-Path $Releases "AutoShotV2_Paper.pdf")
}

Write-Host "Paper build completed: release=$Release"
