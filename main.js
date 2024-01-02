const { app, BrowserWindow, nativeImage, globalShortcut } = require('electron');

const APPICON = "data:image/jpeg;base64,/9j/4QAYRXhpZgAASUkqAAgAAAAAAAAAAAAAAP/sABFEdWNreQABAAQAAABQAAD/4QMqaHR0cDovL25zLmFkb2JlLmNvbS94YXAvMS4wLwA8P3hwYWNrZXQgYmVnaW49Iu+7vyIgaWQ9Ilc1TTBNcENlaGlIenJlU3pOVGN6a2M5ZCI/PiA8eDp4bXBtZXRhIHhtbG5zOng9ImFkb2JlOm5zOm1ldGEvIiB4OnhtcHRrPSJBZG9iZSBYTVAgQ29yZSA1LjUtYzAxNCA3OS4xNTE0ODEsIDIwMTMvMDMvMTMtMTI6MDk6MTUgICAgICAgICI+IDxyZGY6UkRGIHhtbG5zOnJkZj0iaHR0cDovL3d3dy53My5vcmcvMTk5OS8wMi8yMi1yZGYtc3ludGF4LW5zIyI+IDxyZGY6RGVzY3JpcHRpb24gcmRmOmFib3V0PSIiIHhtbG5zOnhtcD0iaHR0cDovL25zLmFkb2JlLmNvbS94YXAvMS4wLyIgeG1sbnM6eG1wTU09Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC9tbS8iIHhtbG5zOnN0UmVmPSJodHRwOi8vbnMuYWRvYmUuY29tL3hhcC8xLjAvc1R5cGUvUmVzb3VyY2VSZWYjIiB4bXA6Q3JlYXRvclRvb2w9IkFkb2JlIFBob3Rvc2hvcCBDQyAoV2luZG93cykiIHhtcE1NOkluc3RhbmNlSUQ9InhtcC5paWQ6QkJFN0FBODZBNjVCMTFFRUFCMDU5MTEwQTQxMTZGRTciIHhtcE1NOkRvY3VtZW50SUQ9InhtcC5kaWQ6QkJFN0FBODdBNjVCMTFFRUFCMDU5MTEwQTQxMTZGRTciPiA8eG1wTU06RGVyaXZlZEZyb20gc3RSZWY6aW5zdGFuY2VJRD0ieG1wLmlpZDpCQkU3QUE4NEE2NUIxMUVFQUIwNTkxMTBBNDExNkZFNyIgc3RSZWY6ZG9jdW1lbnRJRD0ieG1wLmRpZDpCQkU3QUE4NUE2NUIxMUVFQUIwNTkxMTBBNDExNkZFNyIvPiA8L3JkZjpEZXNjcmlwdGlvbj4gPC9yZGY6UkRGPiA8L3g6eG1wbWV0YT4gPD94cGFja2V0IGVuZD0iciI/Pv/uAA5BZG9iZQBkwAAAAAH/2wCEAAICAgICAgICAgIDAgICAwQDAgIDBAUEBAQEBAUGBQUFBQUFBgYHBwgHBwYJCQoKCQkMDAwMDAwMDAwMDAwMDAwBAwMDBQQFCQYGCQ0LCQsNDw4ODg4PDwwMDAwMDw8MDAwMDAwPDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDP/AABEIACAAIAMBEQACEQEDEQH/xACAAAEAAgMAAAAAAAAAAAAAAAAIBgkCAwUBAAIDAQAAAAAAAAAAAAAAAAMEAQIFABAAAgICAgEDBAEFAAAAAAAAAQIDBAUGEQcSACEUQRMWCDIxcYEiIxEAAQQBAgQFBAMAAAAAAAAAAQARAgMhMVFBYRIE8HGRoSKxwTIT8UIU/9oADAMBAAIRAxEAPwCm7rnTZt12fEYGGjkspJkrUVWviMNEs2RuzSclK1VZCI1ZgrFpJD4RoGkfkLwdMMEhJ9Alt1dp+v7n+wGL6d0jpjUs+iRs+Rxmezdu18mGuObLLma5qkSL7+LQRFCf6Kw9NTFcbOhuDpWBmazN8uyT37R/oxS61xOT7K6zRsVrOPggsZ3r3IXlt5HFOSIrSI7MZJkhk4YN7ho28weFPNBWC4Hr/U7fwVEe5cB8PwP5AjBxy3HDKr+pOE45+noSYKz/AF/z2sYT84XMV3OZt4kwYG4LgpRrDY84MhFJK3sFeJ1ZgPdwhiPKyMrH7OELJgT0Qe+nZXAmGvqlD+s8eq7J2RkNg6St4XrzsTVmUaJru4XHNedfsIbkVW+6kQtafycJIpC+RUFeWPq1dlc8PiJIHEiJPj33UWVzreRjmQDtxIHgP5bJb9pSd04eXN9391YqzBTfW8hr+W12i9ezVv070d6GlasiASLX+0zkh/YsCEPA9/T1ka4xlGqfxEcvhyDhvqsyBlOcJW1jq6j0kO8AY/LqORyfQ40VQ8NsgAk8nj3/AL+sglytxlANPipnO4yTIypBQjsB7bSAsjRxKZXjZV9yH8Qn+fQoAE50RJkgY1U10B4Na3HDHKPJhrcqcUXHMkF375VonEgPKv7Dnke5+v09BrmIS6mO3omJRMgz802e9O7thi69n1iW9Qktbnj6WFFnFZO3J9/E1JGt2ZLtCSZ4UkeR4YlcIpIV+Off0/OYId3dIxh8vJA9ZyB6Aipbdy9H4T9ZaW0DU769vjLWBrUmzX8M1errNiRWFiE3lmMbXnJMIeH/AEj4ZHP3D4Dr6jCvqDiOnU2FFFgnMRLPqzqG9HdP5juPN1cFR2Otjdp1WG38Xr3LQ/GyQYRM4tUA3h8pYZo4y6J/0j/kYyAT6U7Xuv1XAXDBDCQ/Eng+x0THe9t+2l6TkFzE6gPltwzrgYjpTetx/PJPxzI4zaOtpK7b1rC05pbXxbsjR171SrwrlWkUq8Y/iSpXkNwGo02QmITBDhxzb67hA/0VWwNlZBDsW0B+2cFarvRe2HQ7PZuqWKW+aPjCqbHk8MZflYZ2YoFymPsRxWaw8gV8yhj5BXz5HHq5j7qoluv/2Q==";

if (!app.requestSingleInstanceLock()) {
    app.exit(0);
}

function createWindow() {
    const mainWindow = new BrowserWindow({
        icon: nativeImage.createFromDataURL(APPICON),
        width: 480,
        height: 900,
        autoHideMenuBar: true,
        resizable: true,
        alwaysOnTop: false,
        webPreferences: {
            nodeIntegration: false
        }
    });

    mainWindow.loadFile('src/mdx.html');

    const reload = ()=> mainWindow.reload();
    const devTools = ()=> mainWindow.webContents.toggleDevTools();
    const register = ()=> {
        globalShortcut.register('f5', reload);
        globalShortcut.register('f1', devTools);
    };
    const unregister = ()=> {
        globalShortcut.unregister('f5', reload);
        globalShortcut.unregister('f1', devTools);
    }
    mainWindow.on('focus', register);
    mainWindow.on('blur', unregister);
    mainWindow.on('beforeunload', unregister);
    mainWindow.on('closed', () => {});

    mainWindow.removeMenu();
}

app.whenReady().then(() => {
    createWindow();

    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0)
            createWindow();
    });
});

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin')
        app.quit();
});
