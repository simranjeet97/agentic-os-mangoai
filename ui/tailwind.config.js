/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: {
          base: '#08090b',
          surface: '#111318',
          elevated: '#1a1d26',
          overlay: '#212432',
        },
        accent: {
          primary: '#6366f1',
          glow: '#818cf8',
          green: '#22c55e',
          red: '#ef4444',
          yellow: '#eab308',
          cyan: '#06b6d4',
          orange: '#f97316',
        },
        border: 'rgba(99, 102, 241, 0.15)',
        'border-hover': 'rgba(99, 102, 241, 0.4)',
        glass: 'rgba(17, 19, 24, 0.7)',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      boxShadow: {
        glow: '0 0 20px rgba(99, 102, 241, 0.3)',
        card: '0 4px 24px rgba(0, 0, 0, 0.6)',
      }
    },
  },
  plugins: [],
}
