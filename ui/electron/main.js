/**
 * electron/main.js — Electron main process for Agentic AI OS desktop app.
 * Creates the BrowserWindow and manages app lifecycle.
 */

const { app, BrowserWindow, shell, ipcMain } = require('electron')
const path = require('path')

const isDev = process.env.NODE_ENV === 'development'
const VITE_PORT = process.env.VITE_PORT || 5173

function createWindow() {
  const win = new BrowserWindow({
    width: 1440,
    height: 900,
    minWidth: 1024,
    minHeight: 700,
    titleBarStyle: 'hiddenInset',
    backgroundColor: '#0a0b0f',
    icon: path.join(__dirname, 'icon.png'),
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
    show: false,
  })

  // Open external links in default browser
  win.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url)
    return { action: 'deny' }
  })

  if (isDev) {
    win.loadURL(`http://localhost:${VITE_PORT}`)
    win.webContents.openDevTools({ mode: 'detach' })
  } else {
    win.loadFile(path.join(__dirname, '../dist/index.html'))
  }

  win.once('ready-to-show', () => win.show())
}

app.whenReady().then(() => {
  createWindow()
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})

// IPC: open file dialog
ipcMain.handle('open-file-dialog', async () => {
  const { dialog } = require('electron')
  const result = await dialog.showOpenDialog({ properties: ['openFile'] })
  return result.filePaths
})
