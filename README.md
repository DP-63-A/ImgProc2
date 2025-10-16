# Version requirements
numpy==2.2.2

opencv-python==4.12.0.88

matplotlib==3.10.6

Pillow==11.3.0

PySide6==6.9.3

jupyterlab==4.4.9

PySimpleGUI==5.0.8.3

These were put into requirements.txt file

# Launching
Create a new folder on C disk and src folder in it, place two .py files into src, then open Windows PowerShell and navigate to project root folder.

Create virtual environment (for Visual Studio Code/Windows PowerShell, execute in project root folder)

python -m venv venv

.\venv\Scripts\activate

py -m pip install --upgrade pip

py -m pip install -r requirements.txt

Execute 'py src\gui.py' to launch the app.
