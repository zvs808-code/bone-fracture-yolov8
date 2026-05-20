// postinstall: node_modules/onnxruntime-web/dist/ → www/ort/ 로 필수 파일 복사.
// npm install 후 자동 실행 (package.json postinstall 스크립트).
const fs = require('fs');
const path = require('path');

const SRC = path.resolve(__dirname, '..', 'node_modules', 'onnxruntime-web', 'dist');
const DST = path.resolve(__dirname, '..', 'www', 'ort');

if (!fs.existsSync(SRC)) {
  console.warn(`⚠️  onnxruntime-web not installed yet — run: npm install`);
  process.exit(0);
}
fs.mkdirSync(DST, { recursive: true });

const NEEDED = [
  'ort.min.js',
  'ort-wasm-simd-threaded.jsep.wasm',
  'ort-wasm-simd-threaded.wasm',
  'ort-wasm-simd-threaded.mjs',
];
let copied = 0;
for (const f of NEEDED) {
  const src = path.join(SRC, f);
  if (fs.existsSync(src)) { fs.copyFileSync(src, path.join(DST, f)); copied++; }
  else console.warn(`(skip missing) ${f}`);
}
if (copied < NEEDED.length) {
  for (const f of fs.readdirSync(SRC)) {
    if (f.endsWith('.wasm') || f.endsWith('.mjs') || f === 'ort.min.js')
      fs.copyFileSync(path.join(SRC, f), path.join(DST, f));
  }
}
console.log(`✅ ORT assets → www/ort/ (${copied} files)`);
