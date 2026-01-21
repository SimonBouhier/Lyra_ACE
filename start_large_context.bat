@echo off
REM ============================================================================
REM LYRA CLEAN - LARGE CONTEXT MODE (32k tokens)
REM ============================================================================
REM Lance Lyra Clean avec une fenêtre de contexte élargie
REM Requiert ~24GB de mémoire GPU/RAM selon le modèle
REM ============================================================================

set "LYRA_MODEL=llama3.1:8b"
set "LYRA_NUM_CTX=32768"

echo.
echo  LYRA CLEAN - Mode Grand Contexte
echo  Modèle: %LYRA_MODEL%
echo  Contexte: %LYRA_NUM_CTX% tokens (~32k)
echo.
echo  Note: Requiert plus de mémoire et augmente la latence
echo.

call "%~dp0start_server.bat"
