# YOLO (ONNXRuntime Web) · análise de imagens no navegador

Este projeto demonstra como executar modelos YOLO exportados para ONNX diretamente no navegador utilizando **onnxruntime-web** com aceleração **WebGPU** (fallback para WASM). Em vez de usar a câmera continuamente, a aplicação permite fazer upload de fotos ou capturá-las na hora e gera um histórico com as imagens já anotadas.

## Recursos
- **Processamento local**: nenhuma foto sai do dispositivo; toda a inferência acontece no browser.
- **Upload ou captura instantânea**: escolha imagens existentes ou use a câmera (`capture=environment`) em um clique.
- **Histórico com anotações**: cada análise salva uma miniatura com as detecções e contagem por classe.
- **Interface em Next.js 14 + Tailwind**: componentes modernos e responsivos.

## Pré-requisitos
- Node.js 18+
- Navegador baseado em Chromium (Chrome/Edge) recente. Ative WebGPU em `chrome://flags/#enable-unsafe-webgpu` se necessário.
- Modelo YOLO exportado para ONNX (`best.onnx`).

## Como rodar
1. Copie seu arquivo `best.onnx` para `public/models/best.onnx`.
2. Instale as dependências:
   ```bash
   npm install
   ```
3. Inicie o ambiente de desenvolvimento:
   ```bash
   npm run dev
   ```
4. Acesse http://localhost:3000 e:
   - Clique em **Escolher da galeria** para selecionar uma imagem existente, ou
   - Clique em **Capturar agora** para tirar uma foto com a câmera.

## Personalização
- Ajuste as classes conhecidas em `components/DetectorONNX.tsx` (`CLASS_NAMES`).
- Controle o tamanho de entrada do modelo (320–640) e o limiar de confiança pela interface.
- O histórico salva até 12 imagens mais recentes por sessão; ajuste `HISTORY_LIMIT` se precisar.

## Notas técnicas
- A aplicação espera saídas no formato `[1, N, 5 + nc]` (padrão YOLO Ultralytics). Se vier `[1, 5 + nc, N]`, o código faz a transposição.
- Para fallback WASM configuramos `SIMD` e múltiplas threads (quando disponíveis).
- Os arquivos gerados (node_modules, `.next/`, etc.) já estão listados no `.gitignore`.

Divirta-se experimentando seu modelo YOLO diretamente no navegador! Se quiser estender o projeto (ex.: exportar relatórios, integrar com back-end, trocar o modelo), use esta base como ponto de partida.
