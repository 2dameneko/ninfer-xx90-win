@echo off
rem Run the nvfp4 Qwen3.8-27B artifact on RTX 5090 as an OpenAI-compatible
rem HTTP server. Conservative profile: 64K context, nvfp4 packed KV,
rem DFlash2 speculative decoding (7 draft tokens).
rem Run download-model.bat (choice 2) once before the first start.
rem On multi-GPU hosts CUDA device numbers can differ from nvidia-smi order;
rem set CUDA_VISIBLE_DEVICES if device 0 is not the card you want.
setlocal
set "ROOT=%~dp0"
set "SERVER=%ROOT%ninfer-serve.exe"
set "MODEL=%ROOT%models\qwen3_8_27b_nvfp4.ninfer"

if not exist "%SERVER%" (
  echo Missing %SERVER% - unpack the whole release archive into one folder first.
  exit /b 1
)
if not exist "%MODEL%" (
  echo Missing %MODEL% - run download-model.bat first and choose option 2.
  exit /b 1
)

echo Starting http://127.0.0.1:8080/v1  (use model id "qwen3.8-27b" in requests)
echo First start loads the weights and takes a couple of minutes.
"%SERVER%" "%MODEL%" --host 127.0.0.1 --port 8080 --device 0 --max-context 65536 --kv-capacity auto --kv-dtype nvfp4 --prefill-chunk 1024 --spec dflash2 --draft-tokens 7
endlocal
