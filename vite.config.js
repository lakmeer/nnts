import { defineConfig } from 'vite'

export default defineConfig({
  define: {
    VITE: true
  },
  resolve: {
    alias: {
      '$src': '/src',
      "$utils": "/src/utils",

      // Deno polymorphic imports
      "npm:ascii-table3": "ascii-table3",

    }
  }
})
