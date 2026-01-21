@echo off
REM ============================================================================
REM LYRA CLEAN - GRANITE 3.3 (4.9GB)
REM ============================================================================
REM Lance Lyra Clean avec IBM Granite 3.3
REM Modèle enterprise, bon pour le code et l'analyse
REM ============================================================================

set "LYRA_MODEL=granite3.3:latest"
echo.
echo  LYRA CLEAN - Modèle: %LYRA_MODEL%
echo.

call "%~dp0start_server.bat"
