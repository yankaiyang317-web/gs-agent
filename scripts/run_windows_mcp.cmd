@echo off
setlocal

rem Local Windows launcher only. Linux server deployment should call
rem ``python -m gs_mcp.server`` in its own CUDA/GCC environment instead.
set "PROJECT_ROOT=%~dp0.."
set "SCENE_MANIFEST=%~1"
if not defined SCENE_MANIFEST set "SCENE_MANIFEST=%GS_SCENE_MANIFEST%"
if not defined SCENE_MANIFEST (
  >&2 echo Missing scene manifest. Pass one as the first argument or set GS_SCENE_MANIFEST.
  exit /b 2
)
if not exist "%PROJECT_ROOT%\%SCENE_MANIFEST%" if not exist "%SCENE_MANIFEST%" (
  >&2 echo Scene manifest not found: %SCENE_MANIFEST%
  exit /b 2
)

call "D:\Users\yankaiyang\anaconda3\condabin\conda.bat" activate gs-agent >nul
if errorlevel 1 exit /b %errorlevel%

rem MCP stdio reserves stdout for JSON-RPC; suppress setup banners accordingly.
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul
if errorlevel 1 exit /b %errorlevel%

set "CUDA_HOME=%CONDA_PREFIX%"
set "CUDA_PATH=%CONDA_PREFIX%"
set "MSVC_BIN=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64"
set "PATH=%MSVC_BIN%;%CONDA_PREFIX%\bin;%PATH%"
set "CUDAHOSTCXX=%MSVC_BIN%\cl.exe"
set "INCLUDE=%CONDA_PREFIX%\Library\include\targets\x64;%CONDA_PREFIX%\Library\include\targets\x64\cccl;%INCLUDE%"
set "LIB=%CONDA_PREFIX%\Library\lib\x64;%LIB%"
set "TORCH_EXTENSIONS_DIR=%PROJECT_ROOT%\.cache\torch_extensions"
set "TORCH_CUDA_ARCH_LIST=8.6"

for %%I in ("%SCENE_MANIFEST%") do set "SCENE_NAME=%%~nI"
if not defined TARGET_VIDEO_CLIPS set "TARGET_VIDEO_CLIPS=1"

pushd "%PROJECT_ROOT%"
python -m gs_mcp.server --scene-manifest "%SCENE_MANIFEST%" --exploration-campaign "outputs\exploration_runs\%SCENE_NAME%" --target-video-clips %TARGET_VIDEO_CLIPS%
set "SERVER_EXIT=%ERRORLEVEL%"
popd
exit /b %SERVER_EXIT%
