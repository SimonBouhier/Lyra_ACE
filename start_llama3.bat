@echo off
REM ============================================================================
REM LYRA CLEAN - LLAMA 3.1 8B (4.9GB)
REM ============================================================================
REM Lance Lyra Clean avec Meta Llama 3.1 8B
REM Modèle polyvalent avec bon équilibre performance/qualité
REM ============================================================================

set "LYRA_MODEL=llama3.1:8b"
echo.
echo  LYRA CLEAN - Modèle: %LYRA_MODEL%
echo.

call "%~dp0start_server.bat"
