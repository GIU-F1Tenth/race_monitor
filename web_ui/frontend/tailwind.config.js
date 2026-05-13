/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'race-bg':        '#0a0a0a',
        'race-panel':     '#111111',
        'race-border':    '#1e1e1e',
        'race-red':       '#e8000d',
        'race-red-glow':  '#ff1a25',
        'race-orange':    '#ff6b00',
        'race-green':     '#00e676',
        'race-cyan':      '#00d4ff',
        'race-yellow':    '#ffd600',
        'race-text':      '#e8e8e8',
        'race-muted':     '#666666',
      },
      fontFamily: {
        mono: ['"Courier New"', 'Courier', 'monospace'],
        ui:   ['Inter', 'system-ui', 'sans-serif'],
      },
      animation: {
        'pulse-red': 'pulse-red 1.5s ease-in-out infinite',
        'blink':     'blink 1s step-end infinite',
      },
      keyframes: {
        'pulse-red': {
          '0%, 100%': { opacity: '1', boxShadow: '0 0 8px #e8000d' },
          '50%':      { opacity: '0.6', boxShadow: '0 0 20px #e8000d, 0 0 40px #e8000d' },
        },
        blink: {
          '0%, 100%': { opacity: '1' },
          '50%':      { opacity: '0' },
        },
      },
    },
  },
  plugins: [],
}
