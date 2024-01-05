const { app, BrowserWindow, nativeImage, globalShortcut } = require('electron');

const APPICON = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAA2lpVFh0WE1MOmNvbS5hZG9iZS54bXAAAAAAADw/eHBhY2tldCBiZWdpbj0i77u/IiBpZD0iVzVNME1wQ2VoaUh6cmVTek5UY3prYzlkIj8+IDx4OnhtcG1ldGEgeG1sbnM6eD0iYWRvYmU6bnM6bWV0YS8iIHg6eG1wdGs9IkFkb2JlIFhNUCBDb3JlIDUuNS1jMDE0IDc5LjE1MTQ4MSwgMjAxMy8wMy8xMy0xMjowOToxNSAgICAgICAgIj4gPHJkZjpSREYgeG1sbnM6cmRmPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5LzAyLzIyLXJkZi1zeW50YXgtbnMjIj4gPHJkZjpEZXNjcmlwdGlvbiByZGY6YWJvdXQ9IiIgeG1sbnM6eG1wTU09Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC9tbS8iIHhtbG5zOnN0UmVmPSJodHRwOi8vbnMuYWRvYmUuY29tL3hhcC8xLjAvc1R5cGUvUmVzb3VyY2VSZWYjIiB4bWxuczp4bXA9Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC8iIHhtcE1NOk9yaWdpbmFsRG9jdW1lbnRJRD0ieG1wLmRpZDpjY2FhOTMxYi1hZDA5LTkzNDEtOTcyYy1mYmY1ODE3ZGFiNzQiIHhtcE1NOkRvY3VtZW50SUQ9InhtcC5kaWQ6OTExRUYxN0RBQkM5MTFFRTgyQzFGRUJERDI2QzgwODQiIHhtcE1NOkluc3RhbmNlSUQ9InhtcC5paWQ6OTExRUYxN0NBQkM5MTFFRTgyQzFGRUJERDI2QzgwODQiIHhtcDpDcmVhdG9yVG9vbD0iQWRvYmUgUGhvdG9zaG9wIENDIChXaW5kb3dzKSI+IDx4bXBNTTpEZXJpdmVkRnJvbSBzdFJlZjppbnN0YW5jZUlEPSJ4bXAuaWlkOjNDQUFDQTVCNjZFNTExRUVCNjFFQTVCQjE4N0EzMTM0IiBzdFJlZjpkb2N1bWVudElEPSJ4bXAuZGlkOjNDQUFDQTVDNjZFNTExRUVCNjFFQTVCQjE4N0EzMTM0Ii8+IDwvcmRmOkRlc2Ny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";

if (!app.requestSingleInstanceLock()) {
    app.exit(0);
}

function createWindow() {
    const mainWindow = new BrowserWindow({
        icon: nativeImage.createFromDataURL(APPICON),
        width: 530,
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
