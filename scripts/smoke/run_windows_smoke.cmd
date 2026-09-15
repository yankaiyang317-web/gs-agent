@echo off
setlocal

rem Activate Conda first: activation may reset PATH. Load MSVC afterwards so
rem it remains visible to PyTorch's JIT compiler.
call "D:\Users\yankaiyang\anaconda3\condabin\conda.bat" activate gs-agent
if errorlevel 1 exit /b %errorlevel%

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64
if errorlevel 1 exit /b %errorlevel%

set "CUDA_HOME=%CONDA_PREFIX%"
set "CUDA_PATH=%CONDA_PREFIX%"
rem Explicitly prepend cl.exe. This avoids a PATH/Path casing conflict inherited
rem from some PowerShell sessions that otherwise hides MSVC from nvcc.
set "MSVC_BIN=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64"
set "PATH=%MSVC_BIN%;%CONDA_PREFIX%\bin;%PATH%"
set "CUDAHOSTCXX=%MSVC_BIN%\cl.exe"
set "INCLUDE=%CONDA_PREFIX%\Library\include\targets\x64;%CONDA_PREFIX%\Library\include\targets\x64\cccl;%INCLUDE%"
set "LIB=%CONDA_PREFIX%\Library\lib\x64;%LIB%"
set "TORCH_EXTENSIONS_DIR=D:\shixi\gs-agent\.cache\torch_extensions"
set "TORCH_CUDA_ARCH_LIST=8.6"

where cl || exit /b 1
where nvcc || exit /b 1

cd /d "D:\shixi\gs-agent"
python scripts\smoke\test_render_pose.py --ply test_data\point_cloud_29999.ply --position 0 0 0 --width 320 --height 240 --out outputs\real_ply_smoke
