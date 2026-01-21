@echo off
REM ============================================================================
REM LYRA CLEAN - MISTRAL (4.4GB)
REM ============================================================================
REM Lance Lyra Clean avec Mistral 7B
REM Excellent rapport qualité/vitesse, multilingue
REM ============================================================================

set "LYRA_MODEL=mistral:latest"
echo.
echo  LYRA CLEAN - Modèle: %LYRA_MODEL%
echo.

call "%~dp0start_server.bat"
