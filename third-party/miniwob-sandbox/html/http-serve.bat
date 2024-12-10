@echo off
set PORT=%1
if "%PORT%"=="" set PORT=8080
echo Serving on port %PORT%
twistd -no web --port tcp:%PORT% --path .