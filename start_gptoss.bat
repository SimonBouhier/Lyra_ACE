@echo off
REM ============================================================================
REM LYRA CLEAN - GPT-OSS 20B (13GB) [DÉFAUT]
REM ============================================================================
REM Lance Lyra Clean avec GPT-OSS 20B
REM Modèle le plus puissant, requiert plus de mémoire
REM ============================================================================

set "LYRA_MODEL=gpt-oss:20b"
echo.
echo  LYRA CLEAN - Modèle: %LYRA_MODEL%
echo.

call "%~dp0start_server.bat"
