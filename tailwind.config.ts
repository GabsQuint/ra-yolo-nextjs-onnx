import type { Config } from 'tailwindcss';

const config: Config = {
  content: [
    './app/**/*.{js,ts,jsx,tsx}',
    './components/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        background: '#030712',
        surface: '#0b1220',
        accent: {
          DEFAULT: '#2563eb',
          emphasis: '#60a5fa',
        },
        text: {
          primary: '#f8fafc',
          muted: '#cbd5f5',
          subtle: '#9ca5c5',
        },
        border: '#1f2937',
      },
      fontFamily: {
        display: ['var(--font-display)', 'Inter', 'system-ui', 'sans-serif'],
        mono: [
          'ui-monospace',
          'SFMono-Regular',
          'Menlo',
          'Monaco',
          'Consolas',
          '"Liberation Mono"',
          '"Courier New"',
          'monospace',
        ],
      },
      boxShadow: {
        focus: '0 0 0 3px rgba(37,99,235,0.35)',
        card: '0 4px 24px rgba(15,23,42,0.45)',
      },
      keyframes: {
        pulseSoft: {
          '0%, 100%': { opacity: '0.12' },
          '50%': { opacity: '0.28' },
        },
      },
      animation: {
        pulseSoft: 'pulseSoft 2s ease-in-out infinite',
      },
    },
  },
  plugins: [],
};

export default config;
