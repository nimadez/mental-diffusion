@echo off
rd /S /Q .output >nul 2>&1

call test_batch.bat
call test_lora.bat
call test_metadata.bat
call test_sd.bat
call test_sdxl.bat
call test_upscale.bat
call test_upscale_no_esrgan.bat
call test_upscale_standalone.bat
call test_vae.bat

pause
