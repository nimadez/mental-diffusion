@echo off
title Mental Diffusion

set CHROME="%ProgramFiles%\Google\Chrome\Application\chrome.exe"
set COMMON=--user-data-dir="%appdata%\chrome-md" --window-size=1400,1000 --window-position=250,50 --no-first-run --disable-plugins --disable-default-apps --disable-extensions --disable-notifications --disable-file-system --disable-background-networking --disable-sync

start "" %CHROME% %COMMON% --app=http://localhost:8011
