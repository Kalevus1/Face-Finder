@echo off
REM Lanzador de Face Finder.
REM Usa el entorno local .venv (creado por instalar.bat) o, si no existe,
REM el entorno .venv_face del autor.
cd /d "%~dp0"

if exist ".venv\Scripts\pythonw.exe" (
  ".venv\Scripts\pythonw.exe" "buscador_de_personas.py"
) else if exist "..\.venv_face\Scripts\pythonw.exe" (
  "..\.venv_face\Scripts\pythonw.exe" "buscador_de_personas.py"
) else (
  echo No se encontro el entorno de Python.
  echo Ejecuta primero "instalar.bat".
  pause
)
