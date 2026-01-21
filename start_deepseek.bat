@echo off
REM ============================================================================
REM LYRA CLEAN - DEEPSEEK-R1 (5.2GB)
REM ============================================================================
REM Lance Lyra Clean avec DeepSeek R1
REM Modèle de raisonnement avancé
REM ============================================================================

set "LYRA_MODEL=deepseek-r1:latest"
echo.
echo  LYRA CLEAN - Modèle: %LYRA_MODEL%
echo.

call "%~dp0start_server.bat"
