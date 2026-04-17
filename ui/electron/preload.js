/**
 * electron/preload.js — Electron context bridge preload script.
 * Exposes a safe API surface to the renderer process.
 */

const { contextBridge, ipcRenderer } = require('electron')

contextBridge.exposeInMainWorld('agenticElectron', {
  openFile: () => ipcRenderer.invoke('open-file-dialog'),
  platform: process.platform,
  versions: {
    node: process.versions.node,
    electron: process.versions.electron,
    chrome: process.versions.chrome,
  },
})
