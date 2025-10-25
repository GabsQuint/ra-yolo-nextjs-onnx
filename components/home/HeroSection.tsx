'use client';

const highlights = [
  {
    title: 'Inferência local e privada',
    description:
      'Carregue o modelo ONNX diretamente da pasta public/ e execute tudo no navegador, preservando seus dados sensíveis.',
  },
  {
    title: 'Aceleração automática',
    description:
      'Selecionamos WebGPU sempre que disponível e caímos para WebAssembly com SIMD e múltiplas threads para manter o fluxo fluido.',
  },
  {
    title: 'Ferramentas para diagnóstico',
    description:
      'Painéis de FPS, confiança mínima e mensagens de status ajudam a ajustar o desempenho e validar o modelo treinado.',
  },
];

export default function HeroSection() {
  return (
    <section className="grid gap-10 lg:grid-cols-[1.05fr_minmax(0,0.95fr)] lg:items-center">
      <div className="space-y-8">
        <div className="inline-flex items-center gap-2 rounded-full border border-accent/40 bg-accent/10 px-4 py-1 text-xs font-semibold uppercase tracking-[0.28em] text-accent">
          ONNX + WebGPU
        </div>
        <div className="space-y-5">
          <h1 className="text-4xl font-semibold leading-tight text-text-primary md:text-5xl">
            Plataforma de visão computacional totalmente no navegador
          </h1>
          <p className="max-w-xl text-base leading-relaxed text-text-subtle">
            Conecte a câmera, carregue o seu modelo YOLO convertido para ONNX e explore resultados em
            realidade aumentada sem back-end ou plugins. Ideal para demonstrações, inspeções em campo e
            provas de conceito rápidas.
          </p>
        </div>
        <ul className="grid gap-5 sm:grid-cols-3">
          {highlights.map((item) => (
            <li key={item.title} className="rounded-2xl border border-border/60 bg-surface/70 p-5 shadow-card">
              <h3 className="text-sm font-semibold text-text-primary">{item.title}</h3>
              <p className="mt-2 text-xs leading-relaxed text-text-subtle">{item.description}</p>
            </li>
          ))}
        </ul>
      </div>
      <div className="relative overflow-hidden rounded-3xl border border-border/60 bg-surface/60 p-6 shadow-card">
        <div className="absolute -top-20 right-10 h-72 w-72 rounded-full bg-accent/20 blur-3xl" />
        <div className="absolute bottom-10 left-10 h-40 w-40 rounded-full bg-accent-emphasis/20 blur-2xl" />
        <div className="relative space-y-5">
          <div className="rounded-2xl border border-border/60 bg-background/60 p-5">
            <p className="text-sm font-semibold text-text-primary">Checklist rápido</p>
            <p className="mt-2 text-xs leading-relaxed text-text-subtle">
              1. Verifique o arquivo <span className="font-mono">public/models/best.onnx</span>.
              <br />
              2. Atualize o navegador (Chrome ou Edge 123+).
              <br />
              3. Ative permissões da câmera ao iniciar.
            </p>
          </div>
          <div className="rounded-2xl border border-border/60 bg-background/60 p-5">
            <p className="text-sm font-semibold text-text-primary">Por baixo dos panos</p>
            <p className="mt-2 text-xs leading-relaxed text-text-subtle">
              O frontend utiliza Next.js 14, Tailwind CSS e onnxruntime-web. Todo o pipeline roda em
              WebAssembly ou WebGPU, com pré-processamento tipo letterbox e pós-processamento NMS
              idênticos ao pipeline da família YOLO.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}
