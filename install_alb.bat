@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" -arch=amd64
call "%~dp0\yolovenv\Scripts\activate.bat"
pip install albumentations
pause
