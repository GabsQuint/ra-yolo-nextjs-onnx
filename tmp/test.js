const ort = require('onnxruntime-node');
const sharp = require('sharp');

const CLASS_NAMES = [
  'bateria_12v',
  'cmos_battery',
  'cpu',
  'cpu_socket',
  'placa_rede_wifi',
  'ram_slot',
  'sata_or_m2',
];

async function preprocess(path, size = 640) {
  const img = sharp(path);
  const meta = await img.metadata();
  const { width, height } = meta;
  const ratio = Math.min(size / width, size / height);
  const newW = Math.round(width * ratio);
  const newH = Math.round(height * ratio);
  const dx = Math.floor((size - newW) / 2);
  const dy = Math.floor((size - newH) / 2);
  const { data, info } = await img
    .resize(newW, newH)
    .raw()
    .toBuffer({ resolveWithObject: true });
  const channels = info.channels;
  const canvas = new Uint8ClampedArray(size * size * 3);
  for (let y = 0; y < newH; y += 1) {
    const srcRow = y * newW * channels;
    const dstRow = (y + dy) * size * 3;
    for (let x = 0; x < newW; x += 1) {
      const srcIdx = srcRow + x * channels;
      const dstIdx = dstRow + (x + dx) * 3;
      canvas[dstIdx] = data[srcIdx];
      canvas[dstIdx + 1] = data[srcIdx + 1];
      canvas[dstIdx + 2] = channels > 2 ? data[srcIdx + 2] : data[srcIdx];
    }
  }
  const plane = size * size;
  const input = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i += 1) {
    input[i] = canvas[i] / 255;
    input[i + plane] = canvas[i + plane] / 255;
    input[i + 2 * plane] = canvas[i + 2 * plane] / 255;
  }
  return { input, width, height, ratio, dx, dy };
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

(async () => {
  const session = await ort.InferenceSession.create('public/models/best.onnx');
  const prep = await preprocess('bus.jpg');
  const feeds = {};
  const inputName = session.inputNames[0];
  feeds[inputName] = new ort.Tensor('float32', prep.input, [1, 3, 640, 640]);
  const results = await session.run(feeds);
  const output = results[session.outputNames[0]];
  console.log('dims', output.dims);
  let data = output.data;
  const shape = output.dims;
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
  }
  console.log('rows/cols', rows, cols);
  const counts = new Array(CLASS_NAMES.length).fill(0);
  let kept = 0;
  for (let r = 0; r < rows; r += 1) {
    const base = r * cols;
    const clsScores = [];
    let bestIdx = -1;
    let bestScore = 0;
    for (let c = 0; c < CLASS_NAMES.length; c += 1) {
      const score = sigmoid(data[base + 4 + c]);
      clsScores.push(score);
      if (score > bestScore) {
        bestScore = score;
        bestIdx = c;
      }
    }
    if (bestScore < 0.4) continue;
    kept += 1;
    counts[bestIdx] += 1;
  }
  console.log('kept', kept, counts);
})();
