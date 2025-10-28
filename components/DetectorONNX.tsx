'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { ChangeEvent } from 'react';
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

const MODEL_SIZE_OPTIONS = [320, 416, 512, 640] as const;
const HISTORY_LIMIT = 12;

type Detection = { x: number; y: number; w: number; h: number; cls: number; conf: number };
type Source = 'upload' | 'camera';

type HistoryItem = {
  id: string;
  timestamp: number;
  source: Source;
  annotatedUrl: string;
  labelCounts: Record<string, number>;
  dimensions: { width: number; height: number };
  fileName?: string;
  totalDetections: number;
};

const classColors = new Map<string, string>();

function colorFor(cls: string) {
  if (!classColors.has(cls)) {
    const color = COLOR_PALETTE[classColors.size % COLOR_PALETTE.length];
    classColors.set(cls, color);
  }
  return classColors.get(cls)!;
}

function iou(a: Detection, b: Detection) {
  const ax2 = a.x + a.w;
  const ay2 = a.y + a.h;
  const bx2 = b.x + b.w;
  const by2 = b.y + b.h;
  const ix1 = Math.max(a.x, b.x);
  const iy1 = Math.max(a.y, b.y);
  const ix2 = Math.min(ax2, bx2);
  const iy2 = Math.min(ay2, by2);
  const iw = Math.max(0, ix2 - ix1);
  const ih = Math.max(0, iy2 - iy1);
  const inter = iw * ih;
  if (inter <= 0) return 0;
  const union = a.w * a.h + b.w * b.h - inter;
  return union <= 0 ? 0 : inter / union;
}

function nms(dets: Detection[], iouThr = 0.45) {
  dets.sort((a, b) => b.conf - a.conf);
  const result: Detection[] = [];
  dets.forEach((det) => {
    if (!result.some((kept) => iou(det, kept) > iouThr)) {
      result.push(det);
    }
  });
  return result;
}

function loadImageFromFile(file: File): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      URL.revokeObjectURL(url);
      resolve(img);
    };
    img.onerror = (err) => {
      URL.revokeObjectURL(url);
      reject(err);
    };
    img.src = url;
  });
}

function generateId() {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return crypto.randomUUID();
  }
  return `history-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export default function DetectorONNX() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const uploadInputRef = useRef<HTMLInputElement>(null);
  const captureInputRef = useRef<HTMLInputElement>(null);
  const sessionRef = useRef<ort.InferenceSession | null>(null);

  const [engine, setEngine] = useState<'webgpu' | 'wasm' | null>(null);
  const [confThr, setConfThr] = useState(0.35);
  const [modelSize, setModelSize] = useState<typeof MODEL_SIZE_OPTIONS[number]>(640);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [currentCounts, setCurrentCounts] = useState<Record<string, number> | null>(null);
  const [currentSource, setCurrentSource] = useState<Source | null>(null);
  const [currentFileName, setCurrentFileName] = useState<string | undefined>();
  const [currentDimensions, setCurrentDimensions] = useState<{ width: number; height: number } | null>(null);
  const [isModelLoading, setIsModelLoading] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [lastError, setLastError] = useState<string | null>(null);

  const statusMessage = useMemo(() => {
    if (isModelLoading) return 'Carregando modelo ONNX...';
    if (isProcessing) return 'Processando imagem...';
    if (!history.length) return 'Envie ou capture uma imagem para começar.';
    return 'Pronto para analisar novas imagens.';
  }, [history.length, isModelLoading, isProcessing]);

  const ensureSession = useCallback(async () => {
    if (sessionRef.current) {
      return sessionRef.current;
    }
    setIsModelLoading(true);
    try {
      let preferredEP: 'webgpu' | 'wasm' = 'wasm';
      if (typeof navigator !== 'undefined' && (navigator as unknown as { gpu?: unknown })?.gpu) {
        preferredEP = 'webgpu';
      }
      const providers = preferredEP === 'webgpu' ? ['webgpu', 'wasm'] : ['wasm'];

      ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/';
      ort.env.wasm.simd = true;
      if (typeof navigator !== 'undefined' && 'hardwareConcurrency' in navigator) {
        const threads = Math.max(1, Math.min(6, navigator.hardwareConcurrency));
        ort.env.wasm.numThreads = threads;
      }

      const response = await fetch('/models/best.onnx', { cache: 'no-store' });
      if (!response.ok) {
        throw new Error(
          response.status === 404
            ? 'Modelo best.onnx não encontrado em public/models.'
            : `Falha ao baixar o modelo (status ${response.status}).`,
        );
      }
      const buffer = await response.arrayBuffer();
      const session = await ort.InferenceSession.create(new Uint8Array(buffer), {
        executionProviders: providers,
        graphOptimizationLevel: 'all',
      });
      const actualEP = (session as unknown as { executionProvider?: string }).executionProvider;
      const selected = actualEP === 'webgpu' ? 'webgpu' : actualEP === 'wasm' ? 'wasm' : preferredEP;
      setEngine(selected);
      sessionRef.current = session;
      return session;
    } finally {
      setIsModelLoading(false);
    }
  }, []);

  useEffect(() => () => {
      sessionRef.current = null;
    }, []);

  const annotateDetections = useCallback(
    (image: HTMLImageElement, detections: Detection[], counts: Record<string, number>) => {
      const canvas = canvasRef.current;
      if (!canvas) return '';
      canvas.width = image.width;
      canvas.height = image.height;
      const ctx = canvas.getContext('2d');
      if (!ctx) return '';
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
      ctx.lineWidth = Math.max(2, canvas.width / 320);
      ctx.font = `600 ${Math.max(14, canvas.width / 40)}px var(--font-display, 'Inter')`;

      detections.forEach((det) => {
        const clsName = CLASS_NAMES[det.cls];
        const color = colorFor(clsName);
        ctx.strokeStyle = color;
        ctx.fillStyle = `${color}33`;
        ctx.fillRect(det.x, det.y, det.w, det.h);
        ctx.strokeRect(det.x, det.y, det.w, det.h);
        const label = `${clsName} ${(det.conf * 100).toFixed(1)}%`;
        const metrics = ctx.measureText(label);
        const labelHeight = metrics.actualBoundingBoxAscent + metrics.actualBoundingBoxDescent + 10;
        ctx.fillStyle = color;
        ctx.fillRect(det.x, Math.max(0, det.y - labelHeight), metrics.width + 14, labelHeight);
        ctx.fillStyle = '#0b1220';
        ctx.fillText(label, det.x + 6, Math.max(12, det.y - 6));
      });

      // Overlay summary
      const total = Object.values(counts).reduce((sum, value) => sum + value, 0);
      const summaryEntries = Object.entries(counts);
      if (total > 0 && summaryEntries.length > 0) {
        ctx.fillStyle = 'rgba(11, 18, 32, 0.75)';
        ctx.fillRect(16, 16, 220, 20 + summaryEntries.length * 24);
        ctx.fillStyle = '#f8fafc';
        ctx.font = '600 16px var(--font-display, "Inter")';
        ctx.fillText(`Total: ${total}`, 24, 36);
        ctx.font = '500 14px var(--font-display, "Inter")';
        summaryEntries.forEach(([name, value], idx) => {
          ctx.fillText(`${name}: ${value}`, 24, 36 + (idx + 1) * 22);
        });
      }

      return canvas.toDataURL('image/png', 0.92);
    },
    [],
  );

  const runInference = useCallback(
    async (file: File, source: Source) => {
      setLastError(null);
      setIsProcessing(true);
      setCurrentSource(source);
      setCurrentFileName(file.name || undefined);
      try {
        const session = await ensureSession();
        const image = await loadImageFromFile(file);
        setCurrentDimensions({ width: image.width, height: image.height });

        const modelW = modelSize;
        const modelH = modelSize;
        const offscreen = document.createElement('canvas');
        offscreen.width = modelW;
        offscreen.height = modelH;
        const offCtx = offscreen.getContext('2d');
        if (!offCtx) throw new Error('Contexto 2D indisponível para pré-processamento.');
        offCtx.fillStyle = '#000000';
        offCtx.fillRect(0, 0, modelW, modelH);
        const ratio = Math.min(modelW / image.width, modelH / image.height);
        const newW = Math.round(image.width * ratio);
        const newH = Math.round(image.height * ratio);
        const dx = Math.floor((modelW - newW) / 2);
        const dy = Math.floor((modelH - newH) / 2);
        offCtx.drawImage(image, dx, dy, newW, newH);

        const imgData = offCtx.getImageData(0, 0, modelW, modelH).data;
        const size = modelW * modelH;
        const input = new Float32Array(3 * size);
        for (let i = 0; i < size; i += 1) {
          const offset = i * 4;
          input[i] = imgData[offset] / 255;
          input[i + size] = imgData[offset + 1] / 255;
          input[i + 2 * size] = imgData[offset + 2] / 255;
        }

        const inputName = session.inputNames[0];
        const tensor = new ort.Tensor('float32', input, [1, 3, modelH, modelW]);
        const outputs = await session.run({ [inputName]: tensor });
        const outputTensor = outputs[session.outputNames[0]] as ort.Tensor;
        let data = outputTensor.data as Float32Array;
        const shape = outputTensor.dims;
        const expectedCols = CLASS_NAMES.length + 4;
        let rows = 0;
        let cols = 0;
        if (shape.length === 3) {
          const [, dim1, dim2] = shape;
          if (dim2 === expectedCols) {
            rows = dim1;
            cols = dim2;
          } else if (dim1 === expectedCols) {
            rows = dim2;
            cols = dim1;
            const transposed = new Float32Array(rows * cols);
            for (let box = 0; box < rows; box += 1) {
              for (let attr = 0; attr < cols; attr += 1) {
                transposed[box * cols + attr] = data[attr * rows + box];
              }
            }
            data = transposed;
          } else {
            cols = expectedCols;
            rows = data.length / cols;
          }
        } else if (shape.length === 2) {
          rows = shape[0];
          cols = shape[1];
        } else {
          cols = expectedCols;
          rows = data.length / cols;
        }

        const detections: Detection[] = [];
        const scale = 1 / ratio;
        for (let r = 0; r < rows; r += 1) {
          const base = r * cols;
          let bx = data[base + 0];
          let by = data[base + 1];
          let bw = data[base + 2];
          let bh = data[base + 3];
          if (cols < 5) continue;
          let bestIdx = -1;
          let bestScore = 0;
          for (let c = 0; c < CLASS_NAMES.length; c += 1) {
            const score = data[base + 4 + c];
            if (score > bestScore) {
              bestScore = score;
              bestIdx = c;
            }
          }
          const conf = bestScore;
          if (conf < confThr || bestIdx < 0) continue;

          const looksNormalized =
            bx >= 0 && bx <= 1 && by >= 0 && by <= 1 && bw >= 0 && bw <= 1 && bh >= 0 && bh <= 1;
          if (looksNormalized) {
            bx *= modelW;
            by *= modelH;
            bw *= modelW;
            bh *= modelH;
          }

          const width = bw * scale;
          const height = bh * scale;
          const xCenter = (bx - dx) * scale;
          const yCenter = (by - dy) * scale;
          const x = Math.max(0, xCenter - width / 2);
          const y = Math.max(0, yCenter - height / 2);
          const right = Math.min(image.width, x + width);
          const bottom = Math.min(image.height, y + height);
          const adjWidth = Math.max(1, right - x);
          const adjHeight = Math.max(1, bottom - y);
          detections.push({ x, y, w: adjWidth, h: adjHeight, cls: bestIdx, conf });
        }

        const finalDetections = nms(detections);
        const finalCounts: Record<string, number> = {};
        finalDetections.forEach((det) => {
          const clsName = CLASS_NAMES[det.cls];
          finalCounts[clsName] = (finalCounts[clsName] || 0) + 1;
        });

        setCurrentCounts(finalCounts);
        const annotated = annotateDetections(image, finalDetections, finalCounts);
        const timestamp = Date.now();
        const newItem: HistoryItem = {
          id: generateId(),
          timestamp,
          source,
          annotatedUrl: annotated,
          labelCounts: finalCounts,
          dimensions: { width: image.width, height: image.height },
          fileName: file.name,
          totalDetections: finalDetections.length,
        };
        setHistory((prev) => [newItem, ...prev].slice(0, HISTORY_LIMIT));
      } catch (error) {
        console.error('Erro ao processar imagem', error);
        const message =
          error instanceof Error ? error.message : 'Erro desconhecido ao processar a imagem. Consulte o console.';
        setLastError(message);
      } finally {
        setIsProcessing(false);
      }
    },
    [annotateDetections, confThr, ensureSession, modelSize],
  );

  const handleInputChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>, source: Source) => {
      const file = event.target.files?.[0];
      event.target.value = '';
      if (file) {
        void runInference(file, source);
      }
    },
    [runInference],
  );

  const totalDetections = useMemo(() => {
    if (!currentCounts) return 0;
    return Object.values(currentCounts).reduce((acc, val) => acc + val, 0);
  }, [currentCounts]);

  return (
    <section className="space-y-8">
      <div className="rounded-3xl border border-border/60 bg-surface/80 p-6 shadow-card backdrop-blur">
        <div className="flex flex-col gap-6 lg:flex-row lg:items-center lg:justify-between">
          <div className="space-y-1.5">
            <p className="text-xs font-medium uppercase tracking-[0.28em] text-accent-emphasis/70">
              Análise por imagem
            </p>
            <h2 className="text-2xl font-semibold text-text-primary">
              Carregue uma foto e visualize as detecções com YOLO + ONNX
            </h2>
            <p className="max-w-xl text-sm text-text-subtle">{statusMessage}</p>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <button
              type="button"
              onClick={() => uploadInputRef.current?.click()}
              disabled={isProcessing || isModelLoading}
              className="inline-flex items-center gap-2 rounded-full bg-accent px-5 py-2 text-sm font-semibold text-white shadow-lg shadow-blue-900/40 transition hover:bg-accent-emphasis focus:outline-none focus-visible:shadow-focus disabled:cursor-not-allowed disabled:bg-accent/40"
            >
              Escolher da galeria
            </button>
            <button
              type="button"
              onClick={() => captureInputRef.current?.click()}
              disabled={isProcessing || isModelLoading}
              className="inline-flex items-center gap-2 rounded-full border border-border/70 px-5 py-2 text-sm font-semibold text-text-muted transition hover:border-accent/80 hover:text-text-primary focus:outline-none focus-visible:shadow-focus disabled:cursor-not-allowed disabled:border-border/40 disabled:text-text-subtle/70"
            >
              Capturar agora
            </button>
          </div>
        </div>

        <input
          ref={uploadInputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(event) => handleInputChange(event, 'upload')}
        />
        <input
          ref={captureInputRef}
          type="file"
          accept="image/*"
          capture="environment"
          className="hidden"
          onChange={(event) => handleInputChange(event, 'camera')}
        />

        <div className="relative mt-6 overflow-hidden rounded-3xl border border-border/60 bg-black/60">
          <canvas ref={canvasRef} className="block h-full w-full max-h-[60vh] object-contain" />
          {!history.length && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 bg-surface/70 backdrop-blur">
              <div className="rounded-full bg-accent/15 px-4 py-1 text-xs font-medium uppercase tracking-[0.24em] text-accent">
                Nenhuma imagem processada
              </div>
              <p className="max-w-md text-center text-sm text-text-subtle">
                Use os botões acima para enviar uma foto da galeria ou capturar uma nova imagem. O modelo YOLO em ONNX
                roda totalmente no navegador e exibe as detecções aqui.
              </p>
            </div>
          )}
          {lastError && (
            <div className="absolute bottom-4 left-4 right-4 rounded-2xl border border-red-500/40 bg-red-500/10 px-4 py-3 text-sm text-red-200">
              {lastError}
            </div>
          )}
          <div className="pointer-events-none absolute left-4 top-4 flex flex-wrap items-center gap-3">
            <span className="rounded-full border border-border/70 bg-surface/70 px-3 py-1 text-xs font-medium text-text-muted">
              Engine: {engine ?? '—'}
            </span>
            <span className="rounded-full border border-border/70 bg-surface/70 px-3 py-1 text-xs font-medium text-text-muted">
              Fonte: {currentSource ?? '—'}
            </span>
            <span className="rounded-full border border-border/70 bg-surface/70 px-3 py-1 text-xs font-medium text-text-muted">
              {isProcessing ? 'Processando...' : totalDetections > 0 ? `Detecções: ${totalDetections}` : 'Detecções: 0'}
            </span>
          </div>
        </div>

        <div className="mt-6 grid gap-5 md:grid-cols-3">
          <div className="flex flex-col gap-3 rounded-2xl border border-border/60 bg-background/40 p-5">
            <span className="text-xs font-semibold uppercase tracking-[0.28em] text-text-subtle/70">
              Confianca minima
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
            <p className="text-xs leading-relaxed text-text-subtle">
              Ajuste o limiar minimo de confianca para equilibrar precisao e cobertura. Valores menores aceitam mais
              hipoteses, enquanto valores altos filtram falsos positivos.
            </p>
          </div>
          <div className="flex flex-col gap-3 rounded-2xl border border-border/60 bg-background/40 p-5">
            <span className="text-xs font-semibold uppercase tracking-[0.28em] text-text-subtle/70">
              Resolucao do modelo
            </span>
            <div className="space-y-2">
              <select
                value={modelSize}
                onChange={(event) => setModelSize(Number(event.target.value) as typeof MODEL_SIZE_OPTIONS[number])}
                disabled={isProcessing || isModelLoading}
                className="w-full rounded-full border border-border/70 bg-surface/80 px-3 py-2 text-sm text-text-primary transition hover:border-accent/70 focus:border-accent focus:outline-none focus-visible:shadow-focus disabled:cursor-not-allowed disabled:bg-surface/40 disabled:text-text-muted"
              >
                {MODEL_SIZE_OPTIONS.map((size) => (
                  <option key={size} value={size}>
                    {size} x {size}
                  </option>
                ))}
              </select>
              <p className="text-xs leading-relaxed text-text-subtle">
                Resolva o modelo com dimensoes menores para acelerar a inferencia em dispositivos modestos. Ajuste antes
                de iniciar uma nova analise.
              </p>
            </div>
            <p className="text-xs leading-relaxed text-text-subtle">
              Dimensao atual do snapshot: {currentDimensions ? `${currentDimensions.width}x${currentDimensions.height}` : '--'}
            </p>
          </div>
          <div className="flex flex-col gap-3 rounded-2xl border border-border/60 bg-background/40 p-5">
            <span className="text-xs font-semibold uppercase tracking-[0.28em] text-text-subtle/70">
              Resultados atuais
            </span>
            {currentCounts && totalDetections > 0 ? (
              <ul className="space-y-2 text-xs leading-relaxed text-text-subtle">
                {Object.entries(currentCounts).map(([name, value]) => (
                  <li key={name} className="flex items-center justify-between">
                    <span>{name}</span>
                    <span className="rounded-full bg-surface/60 px-2 py-0.5 font-semibold text-text-primary">{value}</span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-xs leading-relaxed text-text-subtle">
                Os totais de classes aparecerão aqui assim que uma imagem for processada.
              </p>
            )}
          </div>
        </div>
      </div>

      {history.length > 0 && (
        <div className="rounded-3xl border border-border/60 bg-surface/60 p-6 shadow-card">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.28em] text-text-subtle/70">
                Histórico de análises
              </p>
              <h3 className="text-lg font-semibold text-text-primary">Últimas imagens processadas</h3>
            </div>
            <p className="text-xs text-text-subtle">
              Os últimos {history.length} resultados ficam salvos localmente nesta sessão.
            </p>
          </div>
          <div className="mt-5 grid gap-4 md:grid-cols-2 xl:grid-cols-3">
            {history.map((item) => (
              <div
                key={item.id}
                className="flex flex-col gap-3 rounded-2xl border border-border/60 bg-background/40 p-4 transition hover:border-accent/60 hover:shadow-card"
              >
                <div className="relative overflow-hidden rounded-xl border border-border/60 bg-black/60">
                  <img src={item.annotatedUrl} alt="Resultado anotado" className="h-48 w-full object-cover" />
                  <div className="absolute left-3 top-3 flex flex-wrap items-center gap-2">
                    <span className="rounded-full bg-surface/80 px-2 py-1 text-[11px] font-semibold text-text-muted">
                      {new Date(item.timestamp).toLocaleTimeString()}
                    </span>
                    <span className="rounded-full bg-surface/80 px-2 py-1 text-[11px] font-semibold text-text-muted">
                      {item.source === 'camera' ? 'Capturada' : 'Upload'}
                    </span>
                  </div>
                </div>
                <div className="text-xs text-text-subtle">
                  <p className="font-semibold text-text-primary">
                    {item.fileName ?? 'Imagem analisada'} · {item.dimensions.width}×{item.dimensions.height}px
                  </p>
                  <p className="mt-1">
                    Detecções: <strong>{item.totalDetections}</strong>
                  </p>
                  <div className="mt-2 flex flex-wrap gap-2">
                    {Object.entries(item.labelCounts)
                      .filter(([, value]) => value > 0)
                      .map(([label, value]) => (
                        <span key={label} className="rounded-full bg-surface/70 px-2 py-1 text-[11px] font-semibold text-text-muted">
                          {label}: {value}
                        </span>
                      ))}
                    {item.totalDetections === 0 && (
                      <span className="rounded-full bg-surface/70 px-2 py-1 text-[11px] font-semibold text-text-muted">
                        Nenhuma detecção acima do limiar
                      </span>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}
