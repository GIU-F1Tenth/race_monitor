/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'race-primary': '#1e40af',
        'race-secondary': '#1f2937',
        'race-accent': '#10b981',
        'race-warning': '#f59e0b',
        'race-danger': '#ef4444',
      }
    },
  },
  plugins: [],
}