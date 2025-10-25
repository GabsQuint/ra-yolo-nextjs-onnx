'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import * as ort from 'onnxruntime-web/webgpu';

const CLASS_NAMES = [
  'bateria_12v',
  'cmos_battery',
  'cpu',
  'cpu_socket',
  'placa_rede_wifi',
  'ram_slot',
  'sata_or_m2',
];

const COLOR_PALETTE = [
  '#60a5fa',
  '#34d399',
  '#facc15',
  '#f472b6',
  '#c084fc',
  '#fb7185',
  '#5eead4',
];

const classColors = new Map<string, string>();

const MODEL_SIZE_OPTIONS = [320, 416, 512, 640] as const;
const FRAME_STRIDE_OPTIONS = [1, 2, 3] as const;

function colorFor(cls: string) {
  let color = classColors.get(cls);
  if (!color) {
    color = COLOR_PALETTE[classColors.size % COLOR_PALETTE.length];
    classColors.set(cls, color);
  }
  return color;
}

type LetterboxMeta = {
  r: number;
  ow: number;
  oh: number;
  dw: number;
  dh: number;
};

function letterbox(img: HTMLVideoElement | HTMLImageElement, newShape: [number, number]): LetterboxMeta {
  const w = img instanceof HTMLVideoElement ? img.videoWidth : img.width;
  const h = img instanceof HTMLVideoElement ? img.videoHeight : img.height;
  const [nw, nh] = newShape;
  const r = Math.min(nw / w, nh / h);
  const ow = Math.round(w * r);
  const oh = Math.round(h * r);
  const padw = nw - ow;
  const padh = nh - oh;
  const dw = padw / 2;
  const dh = padh / 2;
  return { r, ow, oh, dw, dh };
}

type Detection = { x: number; y: number; w: number; h: number; cls: number; conf: number };

function iou(a: Detection, b: Detection) {
  const ax1 = a.x;
  const ay1 = a.y;
  const ax2 = a.x + a.w;
  const ay2 = a.y + a.h;
  const bx1 = b.x;
  const by1 = b.y;
  const bx2 = b.x + b.w;
  const by2 = b.y + b.h;

  const ix1 = Math.max(ax1, bx1);
  const iy1 = Math.max(ay1, by1);
  const ix2 = Math.min(ax2, bx2);
  const iy2 = Math.min(ay2, by2);
  const iw = Math.max(0, ix2 - ix1);
  const ih = Math.max(0, iy2 - iy1);
  const inter = iw * ih;
  if (inter <= 0) return 0;
  const areaA = a.w * a.h;
  const areaB = b.w * b.h;
  const union = areaA + areaB - inter;
  return union <= 0 ? 0 : inter / union;
}

function nms(dets: Detection[], iouThr = 0.45) {
  const out: Detection[] = [];
  dets.sort((a, b) => b.conf - a.conf);
  const used = new Array(dets.length).fill(false);
  for (let i = 0; i < dets.length; i++) {
    if (used[i]) continue;
    const a = dets[i];
    out.push(a);
    for (let j = i + 1; j < dets.length; j++) {
      if (used[j]) continue;
      if (iou(a, dets[j]) > iouThr) {
        used[j] = true;
      }
    }
  }
  return out;
}

type Status = 'idle' | 'loading' | 'running';

export default function DetectorONNX() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const sessionRef = useRef<ort.InferenceSession | null>(null);
  const rafRef = useRef<number>();
  const runningRef = useRef(false);
  const inputBufferRef = useRef<Float32Array | null>(null);
  const tensorRef = useRef<ort.Tensor | null>(null);
  const statusRef = useRef<Status>('idle');
  const confRef = useRef(0.35);
  const frameStrideRef = useRef(1);
  const frameCounterRef = useRef(0);

  const [status, setStatus] = useState<Status>('idle');
  const [confThr, setConfThr] = useState(0.35);
  const [fps, setFps] = useState(0);
  const [engine, setEngine] = useState<'webgpu' | 'wasm' | null>(null);
  const [lastError, setLastError] = useState<string | null>(null);
  const [modelSize, setModelSize] = useState<typeof MODEL_SIZE_OPTIONS[number]>(640);
  const [frameStride, setFrameStride] = useState<typeof FRAME_STRIDE_OPTIONS[number]>(1);

  const modelDims = useMemo<[number, number]>(() => [modelSize, modelSize], [modelSize]);

  const isRunning = status === 'running';
  const isLoading = status === 'loading';

  useEffect(() => {
    statusRef.current = status;
  }, [status]);

  useEffect(() => {
    confRef.current = confThr;
  }, [confThr]);

  useEffect(() => {
    frameStrideRef.current = frameStride;
  }, [frameStride]);

  const stop = useCallback(() => {
    runningRef.current = false;
    if (rafRef.current !== undefined) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = undefined;
    }
    const video = videoRef.current;
    if (video) {
      video.pause();
      video.srcObject = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
  }
  statusRef.current = 'idle';
  setStatus('idle');
  setFps(0);
  frameCounterRef.current = 0;
  tensorRef.current = null;
  inputBufferRef.current = null;
  }, []);

  useEffect(() => {
    return () => {
      stop();
      sessionRef.current = null;
    };
  }, [stop]);

  const statusDescription = useMemo(() => {
    switch (status) {
      case 'loading':
        return 'Carregando modelo e preparando a camera...';
      case 'running':
        return 'Deteccao em execucao continua.';
      default:
        return 'Pronto para iniciar a captura.';
    }
  }, [status]);

  const start = useCallback(async () => {
    if (statusRef.current === 'loading' || runningRef.current) {
      return;
    }

    setLastError(null);
    setFps(0);
    statusRef.current = 'loading';
    setStatus('loading');

    try {
      let preferredEP: 'webgpu' | 'wasm' = 'wasm';
      // @ts-ignore - navigator.gpu é experimental
      if (typeof navigator !== 'undefined' && navigator?.gpu) {
        preferredEP = 'webgpu';
      }
      const providers = preferredEP === 'webgpu' ? ['webgpu', 'wasm'] : ['wasm'];

      ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/';
      ort.env.wasm.simd = true;
      if (typeof navigator !== 'undefined' && navigator.hardwareConcurrency) {
        const threads = Math.min(6, Math.max(1, navigator.hardwareConcurrency));
        ort.env.wasm.numThreads = threads;
      }

      let session = sessionRef.current;
      if (!session) {
        const response = await fetch('/models/best.onnx', { cache: 'no-store' });
        if (!response.ok) {
          const errMsg =
            response.status === 404
              ? 'Modelo best.onnx não encontrado em public/models.'
              : `Falha ao baixar o modelo (status ${response.status}).`;
          throw new Error(errMsg);
        }
        const buffer = await response.arrayBuffer();
        const modelData = new Uint8Array(buffer);
        session = await ort.InferenceSession.create(modelData, {
          executionProviders: providers,
          graphOptimizationLevel: 'all',
        });
        sessionRef.current = session;
      }

      const actualEP = (session as unknown as { executionProvider?: string }).executionProvider;
      const selectedEP =
        actualEP === 'webgpu' ? 'webgpu' : actualEP === 'wasm' ? 'wasm' : preferredEP;
      setEngine(selectedEP);

      const captureWidth = modelSize <= 416 ? 960 : 1280;
      const captureHeight = Math.round((captureWidth * 9) / 16);

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'environment',
          width: { ideal: captureWidth },
          height: { ideal: captureHeight },
        },
        audio: false,
      });
      streamRef.current = stream;

      const video = videoRef.current;
      if (!video) {
        throw new Error('Elemento de vídeo não disponível.');
      }
      video.srcObject = stream;
      await video.play();

      const canvas = canvasRef.current;
      if (!canvas) {
        throw new Error('Elemento de canvas não disponível.');
      }
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        throw new Error('Contexto 2D indisponível.');
      }

      canvas.width = video.videoWidth || 1280;
      canvas.height = video.videoHeight || 720;

      const inputName = session.inputNames[0];
      const [modelH, modelW] = modelDims;

      const tmp = document.createElement('canvas');
      tmp.width = modelW;
      tmp.height = modelH;
      const tmpCtx = tmp.getContext('2d');
      if (!tmpCtx) {
        throw new Error('Não foi possível criar o buffer temporário.');
      }

      inputBufferRef.current = new Float32Array(3 * modelW * modelH);
      tensorRef.current = new ort.Tensor('float32', inputBufferRef.current, [1, 3, modelH, modelW]);

      runningRef.current = true;
      statusRef.current = 'running';
      setStatus('running');
      frameCounterRef.current = 0;

      let frames = 0;
      let last = performance.now();

      const loop = async () => {
        if (!runningRef.current) return;

        frameCounterRef.current += 1;
        const stride = Math.max(1, frameStrideRef.current);
        if (stride > 1 && frameCounterRef.current % stride !== 0) {
          rafRef.current = requestAnimationFrame(loop);
          return;
        }
        tmpCtx.fillStyle = '#000';
        tmpCtx.fillRect(0, 0, modelW, modelH);
        const box = letterbox(video, [modelW, modelH]);
        tmpCtx.drawImage(video, box.dw, box.dh, box.ow, box.oh);

        const { data: imgData } = tmpCtx.getImageData(0, 0, modelW, modelH);
        const inputBuffer = inputBufferRef.current!;
        const size = modelW * modelH;
        for (let i = 0; i < size; i++) {
          const offset = i * 4;
          inputBuffer[i] = imgData[offset] / 255;
          inputBuffer[i + size] = imgData[offset + 1] / 255;
          inputBuffer[i + size * 2] = imgData[offset + 2] / 255;
        }

        const tensor = tensorRef.current!;

        let outputTensor: ort.Tensor;
        try {
          const outputs = await session!.run({ [inputName]: tensor });
          outputTensor = outputs[session!.outputNames[0]] as ort.Tensor;
        } catch (error) {
          console.error('Erro na inferência', error);
          setLastError('Falha ao executar a inferência. Consulte o console para detalhes.');
          stop();
          return;
        }

        let data = outputTensor.data as Float32Array;
        let shape = outputTensor.dims;
        let rows = 0;
        let cols = 0;
        if (shape.length === 3) {
          if (shape[1] > shape[2]) {
            rows = shape[2];
            cols = shape[1];
            const transposed = new Float32Array(rows * cols);
            for (let c = 0; c < cols; c++) {
              for (let r = 0; r < rows; r++) {
                transposed[r * cols + c] = data[c * rows + r];
              }
            }
            data = transposed;
          } else {
            rows = shape[1];
            cols = shape[2];
          }
        } else if (shape.length === 2) {
          rows = shape[0];
          cols = shape[1];
        } else {
          rows = data.length / (CLASS_NAMES.length + 5);
          cols = CLASS_NAMES.length + 5;
        }

        const detections: Detection[] = [];
        const gain = box.r;
        const padX = box.dw;
        const padY = box.dh;
        const threshold = confRef.current;

        for (let r = 0; r < rows; r++) {
          const base = r * cols;
          const bx = data[base + 0];
          const by = data[base + 1];
          const bw = data[base + 2];
          const bh = data[base + 3];
          const obj = data[base + 4];
          if (cols < 6) continue;
          let bestIndex = -1;
          let bestScore = 0;
          for (let c = 0; c < CLASS_NAMES.length; c++) {
            const score = data[base + 5 + c];
            if (score > bestScore) {
              bestScore = score;
              bestIndex = c;
            }
          }
          const conf = obj * bestScore;
          if (conf < threshold) continue;
          if (bestIndex < 0) continue;

          const xCenter = (bx - padX) / gain;
          const yCenter = (by - padY) / gain;
          const width = bw / gain;
          const height = bh / gain;
          const x = Math.max(0, xCenter - width / 2);
          const y = Math.max(0, yCenter - height / 2);
          if (x >= canvas.width || y >= canvas.height) continue;
          const clampedW = Math.min(width, canvas.width - x);
          const clampedH = Math.min(height, canvas.height - y);
          if (clampedW <= 1 || clampedH <= 1) continue;
          detections.push({ x, y, w: clampedW, h: clampedH, cls: bestIndex, conf });
        }

        const keep = nms(detections, 0.45);

        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        ctx.lineWidth = 3;
        ctx.font = '600 15px var(--font-display, "Inter")';

        keep.forEach((det) => {
          const color = colorFor(CLASS_NAMES[det.cls]);
          ctx.strokeStyle = color;
          ctx.fillStyle = `${color}33`;
          ctx.fillRect(det.x, det.y, det.w, det.h);
          ctx.strokeRect(det.x, det.y, det.w, det.h);

          const label = `${CLASS_NAMES[det.cls]} ${(det.conf * 100).toFixed(1)}%`;
          ctx.fillStyle = color;
          const textWidth = ctx.measureText(label).width + 12;
          const labelX = Math.max(8, det.x);
          const labelY = Math.max(20, det.y + 18);

          ctx.fillRect(labelX - 6, labelY - 20, textWidth, 24);
          ctx.fillStyle = '#0b1220';
          ctx.fillText(label, labelX, labelY - 4);
        });

        frames += 1;
        const now = performance.now();
        if (now - last >= 1000) {
          setFps(frames);
          frames = 0;
          last = now;
        }

        rafRef.current = requestAnimationFrame(loop);
      };

      rafRef.current = requestAnimationFrame(loop);
    } catch (error) {
      console.error('Falha ao iniciar a captura', error);
      const message =
        error instanceof Error
          ? error.message
          : 'Erro desconhecido ao iniciar a captura. Confira o console.';
      setLastError(message);
      stop();
    }
  }, [stop]);

  return (
    <section className="space-y-6">
      <div className="rounded-3xl border border-border/60 bg-surface/80 p-6 shadow-card backdrop-blur">
        <div className="flex flex-col gap-5 lg:flex-row lg:items-center lg:justify-between">
          <div className="space-y-1.5">
            <p className="text-xs font-medium uppercase tracking-[0.28em] text-accent-emphasis/70">
              Visão computacional
            </p>
            <h2 className="text-2xl font-semibold text-text-primary">
              Laboratório de inferência em tempo real
            </h2>
            <p className="text-sm text-text-subtle">{statusDescription}</p>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <button
              type="button"
              onClick={start}
              disabled={isRunning || isLoading}
              className="inline-flex items-center gap-2 rounded-full bg-accent px-5 py-2 text-sm font-semibold text-white shadow-lg shadow-blue-900/40 transition hover:bg-accent-emphasis focus:outline-none focus-visible:shadow-focus disabled:cursor-not-allowed disabled:bg-accent/40"
            >
              {isLoading ? 'Preparando...' : 'Iniciar camera'}
            </button>
            <button
              type="button"
              onClick={stop}
              disabled={!isRunning && !isLoading}
              className="inline-flex items-center gap-2 rounded-full border border-border/70 px-5 py-2 text-sm font-semibold text-text-muted transition hover:border-accent/80 hover:text-text-primary focus:outline-none focus-visible:shadow-focus disabled:cursor-not-allowed disabled:border-border/40 disabled:text-text-subtle/70"
            >
              Parar camera
            </button>
          </div>
        </div>

        <div className="relative mt-6 overflow-hidden rounded-3xl border border-border/60 bg-black/40">
          <video ref={videoRef} className="hidden" playsInline muted />
          <canvas ref={canvasRef} className="h-full w-full object-cover" />
          {!isRunning && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 bg-surface/75 backdrop-blur-sm">
              <div className="rounded-full bg-accent/15 px-4 py-1 text-xs font-medium uppercase tracking-[0.24em] text-accent">
                Sessao ociosa
              </div>
              <p className="max-w-sm text-center text-sm text-text-subtle">
                Clique em &ldquo;Iniciar camera&rdquo; para habilitar a captura e executar o modelo
                YOLO convertido para ONNX diretamente no navegador.
              </p>
              {lastError && (
                <p className="max-w-lg rounded-2xl border border-red-500/40 bg-red-500/10 px-4 py-2 text-center text-sm text-red-200">
                  {lastError}
                </p>
              )}
            </div>
          )}

          <div className="pointer-events-none absolute left-4 top-4 flex flex-wrap items-center gap-3">
            <span className="rounded-full border border-border/70 bg-surface/70 px-3 py-1 text-xs font-medium text-text-muted">
              {isRunning ? `FPS: ${fps}` : 'FPS: —'}
            </span>
            <span className="rounded-full border border-border/70 bg-surface/70 px-3 py-1 text-xs font-medium text-text-muted">
              Engine: {engine ?? '—'}
            </span>
            <span className="rounded-full border border-border/70 bg-surface/70 px-3 py-1 text-xs font-medium text-text-muted">
              Estado: {isRunning ? 'Executando' : isLoading ? 'Preparando' : 'Pronto'}
            </span>
          </div>
        </div>

        <div className="mt-6 grid gap-5 md:grid-cols-3">
          <div className="flex flex-col gap-3 rounded-2xl border border-border/60 bg-background/40 p-5">
            <span className="text-xs font-semibold uppercase tracking-[0.28em] text-text-subtle/70">
              Sensibilidade e resolucao
            </span>
            <div>
              <input
                type="range"
                min={0.1}
                max={0.9}
                step={0.05}
                value={confThr}
                onChange={(event) => setConfThr(parseFloat(event.target.value))}
                className="h-2 w-full cursor-pointer appearance-none rounded-full bg-border/70 accent-accent focus:outline-none"
              />
              <div className="mt-2 flex items-center justify-between text-xs text-text-muted">
                <span>0.10</span>
                <span className="rounded-full bg-accent/20 px-3 py-1 font-semibold text-accent">
                  {confThr.toFixed(2)}
                </span>
                <span>0.90</span>
              </div>
            </div>
            <div className="mt-4 space-y-2">
              <label className="text-xs font-semibold uppercase tracking-[0.18em] text-text-subtle/70">
                Entrada do modelo (px)
              </label>
              <select
                value={modelSize}
                onChange={(event) => setModelSize(Number(event.target.value) as typeof MODEL_SIZE_OPTIONS[number])}
                disabled={isRunning || isLoading}
                className="w-full rounded-full border border-border/70 bg-surface/80 px-3 py-2 text-sm text-text-primary transition hover:border-accent/70 focus:border-accent focus:outline-none focus-visible:shadow-focus disabled:cursor-not-allowed disabled:bg-surface/40 disabled:text-text-muted"
              >
                {MODEL_SIZE_OPTIONS.map((size) => (
                  <option key={size} value={size}>
                    {size} x {size}
                  </option>
                ))}
              </select>
              <p className="text-xs leading-relaxed text-text-subtle">
                Resoluções menores reduzem o custo da inferência e ajudam a elevar o FPS, especialmente quando apenas o
                back-end WASM está disponível. Ajuste enquanto o detector estiver parado.
              </p>
            </div>
            <div className="mt-4 space-y-2">
              <label className="text-xs font-semibold uppercase tracking-[0.18em] text-text-subtle/70">
                Processar a cada (frames)
              </label>
              <select
                value={frameStride}
                onChange={(event) =>
                  setFrameStride(Number(event.target.value) as typeof FRAME_STRIDE_OPTIONS[number])
                }
                className="w-full rounded-full border border-border/70 bg-surface/80 px-3 py-2 text-sm text-text-primary transition hover:border-accent/70 focus:border-accent focus:outline-none focus-visible:shadow-focus"
              >
                {FRAME_STRIDE_OPTIONS.map((stride) => (
                  <option key={stride} value={stride}>
                    {stride === 1 ? 'Todos os frames' : `1 a cada ${stride} frames`}
                  </option>
                ))}
              </select>
              <p className="text-xs leading-relaxed text-text-subtle">
                Pule frames intermediários para aliviar a GPU quando o cenário está estável. Com WebGPU, um stride de 2
                costuma dobrar o FPS percebido mantendo boa responsividade.
              </p>
            </div>
            <p className="text-xs leading-relaxed text-text-subtle">
              Ajuste o limiar mínimo de confiança das detecções. Valores maiores filtram falsos positivos; valores
              menores aumentam a sensibilidade.
            </p>
          </div>
          <div className="flex flex-col gap-3 rounded-2xl border border-border/60 bg-background/40 p-5">
            <span className="text-xs font-semibold uppercase tracking-[0.28em] text-text-subtle/70">
              Otimização
            </span>
            <ul className="space-y-2 text-xs leading-relaxed text-text-subtle">
              <li>- WebGPU é selecionado automaticamente quando disponível.</li>
              <li>- Para WebAssembly habilitamos SIMD e múltiplas threads por padrão.</li>
              <li>- Combine a resolução do modelo com um navegador Chromium atualizado para maximizar FPS.</li>
            </ul>
          </div>
          <div className="flex flex-col gap-3 rounded-2xl border border-border/60 bg-background/40 p-5">
            <span className="text-xs font-semibold uppercase tracking-[0.28em] text-text-subtle/70">
              Boas práticas
            </span>
            <ul className="space-y-2 text-xs leading-relaxed text-text-subtle">
              <li>- Confirme que `public/models/best.onnx` foi exportado com suporte a shapes dinâmicos.</li>
              <li>- Iluminação uniforme e estabilidade da câmera ajudam a evitar quedas de confiança.</li>
              <li>- Verifique em chrome://gpu se WebGPU está ativo; atualize o navegador se necessário.</li>
            </ul>
          </div>
        </div>
      </div>
    </section>
  );
}
