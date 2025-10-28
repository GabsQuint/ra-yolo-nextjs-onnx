'use client';

const highlights = [
  {
    title: 'Inferencia local',
    description: 'O modelo ONNX roda inteiramente no navegador. Nenhum upload externo durante a analise.',
  },
  {
    title: 'Upload ou captura',
    description: 'Escolha fotos da galeria ou use a camera do dispositivo via capture=environment em um toque.',
  },
  {
    title: 'Historico anotado',
    description: 'Cada execucao gera miniaturas com contagem por classe para comparar resultados rapidamente.',
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
            Analise fotos com YOLO direto no navegador
          </h1>
          <p className="max-w-xl text-base leading-relaxed text-text-subtle">
            Carregue uma imagem da galeria ou capture uma foto na hora e visualize as deteccoes sobrepostas em segundos.
            Sem back-end, plugins ou instalacao: apenas Next.js, Tailwind e onnxruntime-web.
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
            <p className="text-sm font-semibold text-text-primary">Checklist rapido</p>
            <p className="mt-2 text-xs leading-relaxed text-text-subtle">
              1. Garanta o arquivo <span className="font-mono">public/models/best.onnx</span> exportado do YOLO.
              <br />
              2. Utilize Chrome ou Edge atualizados com WebGPU ativado.
              <br />
              3. Permita acesso a camera apenas se desejar capturar novas imagens.
            </p>
          </div>
          <div className="rounded-2xl border border-border/60 bg-background/60 p-5">
            <p className="text-sm font-semibold text-text-primary">Tecnologia</p>
            <p className="mt-2 text-xs leading-relaxed text-text-subtle">
              Empilhamos Next.js 14, Tailwind CSS e onnxruntime-web. O pre-processamento utiliza letterbox e o
              pos-processamento aplica NMS classico da familia YOLO, tudo no browser.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}
