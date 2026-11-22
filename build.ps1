param([switch]$CpuOnly)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
# Resolve redistributable DLLs from Windows and the build environment.
$env:PATH = "$(Split-Path -Parent $python);$env:SystemRoot\System32;$env:SystemRoot"
$env:PYINSTALLER_CONFIG_DIR = Join-Path $PSScriptRoot 'build\.pyinstaller-cache'
$gpuPackages = Join-Path $PSScriptRoot '.venv\Lib\site-packages\nvidia'
$gpuComponents = 'cublas', 'cuda_nvrtc', 'cuda_runtime', 'cudnn'
$distPath = 'dist'
if ($CpuOnly) {
    $distPath = 'dist\portable'
} else {
    foreach ($component in $gpuComponents) {
        if (-not (Test-Path -LiteralPath (Join-Path $gpuPackages "$component\bin") -PathType Container)) {
            throw 'Instale requirements-build.txt para empacotar o suporte à GPU.'
        }
    }
}

& $python -m PyInstaller --noconfirm --onedir --windowed --name Transcritor `
    --specpath build --workpath 'build\package' --distpath $distPath `
    --exclude-module hf_xet `
    --add-data "$(Join-Path $PSScriptRoot 'theme.qss'):." `
    --collect-all faster_whisper --collect-all huggingface_hub `
    --collect-all sherpa_onnx --collect-all yt_dlp --collect-all yt_dlp_ejs `
    --collect-binaries ctranslate2 `
    --copy-metadata huggingface-hub --copy-metadata sherpa-onnx `
    --copy-metadata sherpa-onnx-core --copy-metadata yt-dlp-ejs `
    app.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
if ($CpuOnly) { return }

$gpuBundle = Join-Path $PSScriptRoot 'dist\Transcritor\_internal\nvidia'
foreach ($component in $gpuComponents) {
    $destination = Join-Path $gpuBundle "$component\bin"
    New-Item -ItemType Directory -Force -Path $destination | Out-Null
    Get-ChildItem -LiteralPath (Join-Path $gpuPackages "$component\bin") -File -Filter '*.dll' |
        Copy-Item -Destination $destination
}
Get-ChildItem -LiteralPath (Split-Path -Parent $gpuPackages) -Directory -Filter 'nvidia_*.dist-info' |
    ForEach-Object {
        $licenses = Join-Path $_.FullName 'licenses'
        if (Test-Path -LiteralPath $licenses) {
            Copy-Item -LiteralPath $licenses -Recurse -Destination (Join-Path $gpuBundle $_.Name)
        }
    }

$shortcut = (New-Object -ComObject WScript.Shell).CreateShortcut((Join-Path $PSScriptRoot 'Abrir Transcritor.lnk'))
$shortcut.TargetPath = Join-Path $PSScriptRoot 'dist\Transcritor\Transcritor.exe'
$shortcut.WorkingDirectory = Join-Path $PSScriptRoot 'dist\Transcritor'
$shortcut.Description = 'Transcritor com Whisper Large-v3 e GPU NVIDIA'
$shortcut.Save()
