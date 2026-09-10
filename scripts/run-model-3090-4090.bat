@echo off
rem Run the groupwise-int Qwen3.8-27B artifact on RTX 3090 / RTX 4090 as an
rem OpenAI-compatible HTTP server. Conservative profile: 32K context,
rem rk8v4 packed KV, DFlash2 speculative decoding (7 draft tokens).
rem Run download-model.bat (choice 1) once before the first start.
rem On multi-GPU hosts CUDA device numbers can differ from nvidia-smi order;
rem set CUDA_VISIBLE_DEVICES if device 0 is not the card you want.
setlocal
set "ROOT=%~dp0"
set "SERVER=%ROOT%ninfer-serve.exe"
set "MODEL=%ROOT%models\qwen3_8_27b.ninfer"

if not exist "%SERVER%" (
  echo Missing %SERVER% - unpack the whole release archive into one folder first.
  exit /b 1
)
if not exist "%MODEL%" (
  echo Missing %MODEL% - run download-model.bat first and choose option 1.
  exit /b 1
)

echo Starting http://127.0.0.1:8080/v1  (use model id "qwen3.8-27b" in requests)
echo First start loads the weights and takes a couple of minutes.
"%SERVER%" "%MODEL%" --host 127.0.0.1 --port 8080 --device 0 --max-context 32768 --kv-capacity 32768 --kv-dtype rk8v4 --prefill-chunk 1024 --spec dflash2 --draft-tokens 7
endlocal
