$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$config = Join-Path ([System.IO.Path]::GetTempPath()) "koharu-nsis-$([guid]::NewGuid()).json"

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
