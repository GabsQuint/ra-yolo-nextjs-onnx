'use client';

const steps = [
  {
    title: 'Prepare o modelo',
    detail: 'Exporte o seu YOLO como ONNX (opset >= 12) e coloque o arquivo em public/models/best.onnx.',
  },
  {
    title: 'Envie ou capture fotos',
    detail:
      'Use o botao de upload para escolher imagens da galeria ou capture uma foto com a camera do dispositivo.',
  },
  {
    title: 'Revise o historico',
    detail:
      'Analise as anotacoes geradas, ajuste o limiar de confianca e compare as miniaturas salvas automaticamente.',
  },
];

export default function HowItWorksSection() {
  return (
    <section className="rounded-3xl border border-border/60 bg-surface/70 p-8 shadow-card">
      <div className="flex flex-col gap-6 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.28em] text-text-subtle/70">
            Como funciona
          </p>
          <h2 className="mt-1 text-2xl font-semibold text-text-primary">
            Três passos para inspecionar suas imagens com YOLO
          </h2>
        </div>
        <p className="max-w-2xl text-sm leading-relaxed text-text-subtle">
          Ideal para validar modelos de visao computacional, criar laudos rapidos ou demonstrar resultados para clientes
          sem depender de servidores dedicados.
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
