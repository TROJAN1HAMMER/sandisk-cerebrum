/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        silicon: {
          canvas: "#080B10",
          surface: "#0E131F",
          card: "#151D2C",
          elevated: "#1C2638",
          interactive: "#212B3D",
          border: "#32435E",
          borderLight: "#475569",
        },
        telemetry: {
          cyan: "#38BDF8",
          green: "#22C55E",
          red: "#EF4444",
          amber: "#F59E0B",
          blue: "#3B82F6",
          purple: "#A855F7",
          analytics: "#06B6D4",
        },
        content: {
          primary: "#F1F5F9",
          secondary: "#94A3B8",
          muted: "#64748B",
        }
      },
      fontFamily: {
        sans: ["Inter", "-apple-system", "BlinkMacSystemFont", "Segoe UI", "Roboto", "sans-serif"],
        mono: ["JetBrains Mono", "SFMono-Regular", "Consolas", "Menlo", "monospace"],
      },
    },
  },
  plugins: [],
}
