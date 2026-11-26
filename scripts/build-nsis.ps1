$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$config = Join-Path ([System.IO.Path]::GetTempPath()) "koharu-nsis-$([guid]::NewGuid()).json"
$cores = (Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum
if (-not $cores -or $cores -lt 1) {
    throw 'Could not determine the number of physical CPU cores.'
}
$env:CARGO_BUILD_JOBS = [string]$cores

$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (Test-Path -LiteralPath $vswhere) {
    $vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if ($vs) {
        $cmakeTools = Join-Path $vs 'Common7\IDE\CommonExtensions\Microsoft\CMake'
        foreach ($directory in @('CMake\bin', 'Ninja')) {
            $path = Join-Path $cmakeTools $directory
            if (Test-Path -LiteralPath $path) {
                $env:PATH = "$path;$env:PATH"
            }
        }
    }
}

try {
    [System.IO.File]::WriteAllText($config, '{"bundle":{"createUpdaterArtifacts":false}}')
    Push-Location $root
    try {
        cargo tauri build --bundles nsis --config $config
        if ($LASTEXITCODE -ne 0) {
            throw "Tauri build failed with exit code $LASTEXITCODE."
        }
    }
    finally {
        Pop-Location
    }
}
finally {
    Remove-Item -LiteralPath $config -Force -ErrorAction SilentlyContinue
}
