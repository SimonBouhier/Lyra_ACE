@echo off
REM ============================================================================
REM LYRA CLEAN - DIAGNOSTIC & SETUP
REM ============================================================================
REM Vérifie les dépendances et configure l'environnement
REM ============================================================================

setlocal enabledelayedexpansion

set "PROJECT_DIR=%~dp0"
set "VENV_PATH=%PROJECT_DIR%.venv"
set "PYTHON_EXE=%VENV_PATH%\Scripts\python.exe"

echo.
echo ============================================================================
echo  LYRA CLEAN - DIAGNOSTIC SYSTÈME
echo ============================================================================
echo.

REM Créer venv si absent
if not exist "%VENV_PATH%" (
    echo 📦 Création de l'environnement virtuel...
    python -m venv "%VENV_PATH%"
    if !errorlevel! neq 0 (
        echo ❌ Erreur: Impossible de créer le venv
        pause
        exit /b 1
    )
    echo ✓ Environnement virtuel créé
    echo.
)

REM Vérifier Python
echo [1/5] Vérification Python...
"%PYTHON_EXE%" --version
if %errorlevel% neq 0 (
    echo ❌ Python n'est pas accessible
    pause
    exit /b 1
)
echo ✓ Python OK
echo.

REM Installer/mettre à jour les dépendances
echo [2/5] Installation des dépendances...
"%PYTHON_EXE%" -m pip install -q -r "%PROJECT_DIR%requirements.txt" 2>nul
if %errorlevel% neq 0 (
    echo ⚠️  Certaines dépendances n'ont pas pu être installées
    echo   Continuant avec les dépendances existantes...
    echo.
)
echo ✓ Dépendances vérifiées
echo.

REM Tester Ollama
echo [3/5] Vérification Ollama...
"%PYTHON_EXE%" -c "import httpx; r = httpx.get('http://localhost:11434/api/tags', timeout=5); print('✓ Ollama disponible'); print('  Modèles: ' + ', '.join([m['name'] for m in r.json().get('models', [])]))" 2>nul
if !errorlevel! neq 0 (
    echo ❌ Ollama n'est pas accessible à http://localhost:11434
    echo    Lancez Ollama avant le serveur
    echo.
) else (
    echo.
)

REM Tester les imports principaux
echo [4/5] Vérification des imports...
"%PYTHON_EXE%" -c "from app.main import app; from services.consciousness import SemanticMemory; print('✓ Tous les imports OK')" 2>nul
if !errorlevel! neq 0 (
    echo ❌ Erreur d'import
    "%PYTHON_EXE%" -c "from app.main import app; from services.consciousness import SemanticMemory"
    pause
    exit /b 1
)
echo.

REM Vérifier la base de données
echo [5/5] Vérification de la base de données...
if exist "%PROJECT_DIR%ispace.db" (
    echo ✓ Base de données trouvée: ispace.db
) else (
    echo ⚠️  Base de données ispace.db non trouvée
    echo    Elle sera créée au premier démarrage
)
echo.

echo ============================================================================
echo  ✅ DIAGNOSTIC TERMINÉ - PRÊT À DÉMARRER
echo ============================================================================
echo.
echo Pour démarrer le serveur, double-cliquez sur START_SERVER.bat
echo.
pause
