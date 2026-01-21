@echo off
REM ============================================================================
REM LYRA CLEAN - GEMMA3 (3.3GB)
REM ============================================================================
REM Lance Lyra Clean avec le modèle Google Gemma 3
REM Modèle léger et rapide, bon pour le développement
REM ============================================================================

set "LYRA_MODEL=gemma3:latest"
echo.
echo  LYRA CLEAN - Modèle: %LYRA_MODEL%
echo.

call "%~dp0start_server.bat"
