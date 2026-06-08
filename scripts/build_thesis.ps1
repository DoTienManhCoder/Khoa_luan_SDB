param(
    [ValidateSet("pdf", "slides", "all")]
    [string]$Target = "all",
    [switch]$Release
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Thesis = Join-Path $Root "publications\thesis"
$Build = Join-Path $Thesis "build"
$Releases = Join-Path $Thesis "releases"

New-Item -ItemType Directory -Force -Path $Build, $Releases | Out-Null

& python (Join-Path $PSScriptRoot "sync_experimental_results.py") --check
if ($LASTEXITCODE -ne 0) {
    throw "Experimental result outputs are stale. Run sync_experimental_results.py --write."
}

if ($Target -in @("pdf", "all")) {
    Push-Location $Thesis
    try {
        & pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
        if ($LASTEXITCODE -ne 0) { throw "First pdflatex pass failed." }

        & biber --input-directory=build --output-directory=build main
        if ($LASTEXITCODE -ne 0) { throw "Biber pass failed." }

        & pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
        if ($LASTEXITCODE -ne 0) { throw "Second pdflatex pass failed." }

        & pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
        if ($LASTEXITCODE -ne 0) { throw "Final pdflatex pass failed." }
    }
    finally {
        Pop-Location
    }

    if ($Release) {
        Copy-Item -Force -LiteralPath (Join-Path $Build "main.pdf") `
            -Destination (Join-Path $Releases "AutoShotV2_Thesis.pdf")
    }
}

if ($Target -in @("slides", "all")) {
    & python (Join-Path $Thesis "slides\build_slides.py") `
        --output (Join-Path $Build "AutoShotV2_Defense.pptx")
    if ($LASTEXITCODE -ne 0) { throw "Slide build failed." }

    if ($Release) {
        Copy-Item -Force -LiteralPath (Join-Path $Build "AutoShotV2_Defense.pptx") `
            -Destination (Join-Path $Releases "AutoShotV2_Defense.pptx")
    }
}

Write-Host "Build completed: target=$Target release=$Release"
