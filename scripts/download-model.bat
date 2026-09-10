@echo off
rem Download the model artifact for ninfer-xx90-win into the .\models folder.
rem Downloads are resumable: re-run this script after a break to continue.
setlocal
set "ROOT=%~dp0"
set "MODELS=%ROOT%models"

echo.
echo Which GPU will this machine use?
echo   1) RTX 3090 / RTX 4090  - groupwise-int artifact, ~19 GB
echo   2) RTX 5090             - nvfp4 artifact, ~22 GB
set "GPU="
set /p "GPU=Choice [1/2]: "

if "%GPU%"=="1" (
  set "REPO=neroued/Qwen3.8-27B-NInfer"
  set "FILE=qwen3_8_27b.ninfer"
)
if "%GPU%"=="2" (
  set "REPO=neroued/Qwen3.8-27B-nvfp4-NInfer"
  set "FILE=qwen3_8_27b_nvfp4.ninfer"
)
if "%FILE%"=="" (
  echo Please answer 1 or 2.
  exit /b 1
)

if not exist "%MODELS%" mkdir "%MODELS%"
if exist "%MODELS%\%FILE%" (
  echo Model already present: %MODELS%\%FILE%
  exit /b 0
)

echo.
echo Downloading https://huggingface.co/%REPO%/resolve/main/%FILE%
echo to %MODELS%\%FILE%
echo This is a large file; progress is shown by curl. If the download is
echo interrupted, run this script again and it will resume.
curl.exe -L -C - --retry 3 --retry-delay 5 -o "%MODELS%\%FILE%" "https://huggingface.co/%REPO%/resolve/main/%FILE%"
if errorlevel 1 (
  echo.
  echo Download did not finish. Run this script again to resume.
  exit /b 1
)
echo Done: %MODELS%\%FILE%
endlocal
