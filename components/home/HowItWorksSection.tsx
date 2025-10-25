'use client';

const steps = [
  {
    title: 'Prepare o modelo',
    detail:
      'Exporte o seu YOLO treinado como ONNX (opset ≥ 12) e copie o arquivo para public/models/best.onnx.',
  },
  {
    title: 'Execute no browser',
    detail:
      'Clique em “Iniciar câmera”, conceda permissão e acompanhe as detecções renderizadas em tempo real sobre o vídeo.',
  },
  {
    title: 'Ajuste a sensibilidade',
    detail:
      'Use os controles de confiança para equilibrar falsos positivos, monitore o FPS e compare WebGPU vs WASM conforme o hardware.',
  },
];

export default function HowItWorksSection() {
  return (
    <section className="rounded-3xl border border-border/60 bg-surface/70 p-8 shadow-card">
      <div className="flex flex-col gap-6 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.28em] text-text-subtle/70">
            Como operar
          </p>
          <h2 className="mt-1 text-2xl font-semibold text-text-primary">
            Três etapas para levar seu modelo de laboratório ao campo
          </h2>
        </div>
        <p className="max-w-2xl text-sm leading-relaxed text-text-subtle">
          Esta aplicação foi pensada para apoiar equipes de P&amp;D e suporte técnico. Ela permite
          validar modelos em linha de produção, fazer troubleshooting remoto ou criar demonstrações
          interativas com total portabilidade.
        </p>
      </div>
      <div className="mt-8 grid gap-6 md:grid-cols-3">
        {steps.map((step, index) => (
          <div key={step.title} className="flex flex-col gap-3 rounded-2xl bg-background/40 p-6">
            <span className="inline-flex h-9 w-9 items-center justify-center rounded-full bg-accent/20 text-sm font-semibold text-accent-emphasis">
              {index + 1}
            </span>
            <p className="text-sm font-semibold text-text-primary">{step.title}</p>
            <p className="text-xs leading-relaxed text-text-subtle">{step.detail}</p>
          </div>
        ))}
      </div>
    </section>
  );
}
