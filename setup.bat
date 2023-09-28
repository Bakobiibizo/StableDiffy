@echo off

REM setup env
python -m venv .venv
echo "setx PYTHONPATH $PYTHONPATH" >> ".venv/scripts/activate.bat"

