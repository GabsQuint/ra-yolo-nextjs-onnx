import './globals.css';
import { Plus_Jakarta_Sans } from 'next/font/google';
import type { Metadata } from 'next';

const font = Plus_Jakarta_Sans({
  subsets: ['latin'],
  weight: ['400', '500', '600', '700'],
  variable: '--font-display',
  display: 'swap',
});

export const metadata: Metadata = {
  title: 'YOLO ONNX Vision Lab',
  description:
    'Detecção em tempo real com YOLO e ONNX Runtime direto no navegador usando WebGPU ou WebAssembly.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="pt-BR" suppressHydrationWarning>
      <body className={`${font.variable} bg-background text-text-primary min-h-screen`}>{children}</body>
    </html>
  );
}
