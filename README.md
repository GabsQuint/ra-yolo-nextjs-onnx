# YOLO (ONNXRuntime Web) – AR no navegador

Detecção em tempo real no navegador (sem backend) usando **onnxruntime-web**.  
Funciona com modelos YOLO exportados para **ONNX** (Ultralytics).

## Passos
1. Copie seu `best.onnx` para `public/models/best.onnx`.
2. Instale dependências:
   ```bash
   npm i
   ```
3. Rode em dev:
   ```bash
   npm run dev
   ```
4. Abra http://localhost:3000, clique em **Iniciar câmera**.

## Notas
- Prioriza **WebGPU** automaticamente, com fallback para **WASM (CPU)**.
- Espera saída com shape `[1, N, 5+nc]` (YOLO Ultralytics). Se vier `[1, 5+nc, N]`, o código transpõe.
- Se seu input não for 640×640, ajuste `modelW/modelH` em `DetectorONNX.tsx`.

## Classes
Edite a lista de classes em `components/DetectorONNX.tsx`:
```ts
const CLASS_NAMES = ['bateria_12v','cmos_battery','cpu','cpu_socket','placa_rede_wifi','ram_slot','sata_or_m2'];
```

## Dicas de performance
- Use WebGPU para dispositivos com suporte (Chrome/Edge recentes).
- Reduza a taxa de inferência (debounce) se precisar economizar CPU.
