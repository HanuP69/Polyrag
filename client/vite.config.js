import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import fs from 'fs'
import path from 'path'

let node_port = 3001
let frontend_port = 5173

try {
  const configPath = path.resolve(__dirname, '../polyrag.config.json')
  if (fs.existsSync(configPath)) {
    const config = JSON.parse(fs.readFileSync(configPath, 'utf8'))
    if (config.server) {
      if (config.server.node_port) node_port = config.server.node_port
      if (config.server.frontend_port) frontend_port = config.server.frontend_port
    }
  }
} catch (err) {
  console.error('Error loading polyrag.config.json in vite.config.js:', err.message)
}

export default defineConfig({
  plugins: [react()],
  server: {
    port: frontend_port,
    proxy: {
      '/api': {
        target: `http://localhost:${node_port}`,
        changeOrigin: true,
      },
    },
  },
})
