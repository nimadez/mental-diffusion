@echo off

echo -------------------
echo  VS Code Symlinker
echo -------------------

:: admin check
net session >nul 2>&1
if not %errorLevel% == 0 (
    echo Run as administrator && pause && exit
)

if exist %USERPROFILE%\.vscode\extensions (
    mklink /D %USERPROFILE%\.vscode\extensions\mental-diffusion %~dp0
    goto END
) else (
    goto LINK
)

:LINK
echo  VSCode extensions directory is not detected.
echo.
echo  Enter path to VSCode extensions directory
echo  (example: D:\Apps\VSC\data\extensions)
echo.
set /p path="Path: "
echo.
mklink /D %path%\mental-diffusion %~dp0
echo.

:END
pause
