@echo off
title Mental Diffusion Client

set CHROME="%ProgramFiles%\Google\Chrome\Application\chrome.exe"
set COMMON=--user-data-dir="%appdata%\mental-diffusion" --window-size=480,900 --window-position=250,50 --no-first-run --disable-plugins --disable-default-apps --disable-extensions --disable-notifications --disable-file-system --disable-background-networking --disable-sync

start "" %CHROME% %COMMON% --app=http://localhost:8011
