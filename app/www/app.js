/*
 * Bone Fracture Detector — on-device ONNX YOLOv8 inference (브라우저/Capacitor/Electron 공용).
 *
 * 입력: <input type=file> 또는 드래그앤드롭 X-ray 이미지
 * 처리: HTMLImageElement → letterbox(640×640) → Float32Array (1,3,640,640) / 255
 * 추론: ort.InferenceSession.run({images}) → output0 [1, 5, N]
 * 후처리: confidence filter → NMS → draw bounding boxes on canvas
 */
(() => {
  'use strict';

  /* ── DOM ─────────────────────────────────────────────── */
  const $ = (s) => document.querySelector(s);
  const dropEl    = $('#drop');
  const fileEl    = $('#fileInput');
  const runEl     = $('#runBtn');
  const statEl    = $('#status');
  const resEl     = $('#result');
  const badgeEl   = $('#modelBadge');
  const infoEl    = $('#modelInfo');
  const confSlider = $('#confSlider');
  const confValEl = $('#confVal');
  const canvasWrap = $('#canvasWrap');
  const srcCanvas = $('#srcCanvas');
  const detCanvas = $('#detCanvas');

  const BOX_COLORS = ['#34A853', '#4285F4', '#FBBC04', '#EA4335', '#7B61FF', '#00BCD4'];

  /* ── 상태 ─────────────────────────────────────────────── */
  let META = null;
  let SESSION = null;
  let lastBitmap = null;
  let lastFileName = '';
  let currentDets = [];

  /* ── 신뢰도 슬라이더 ─────────────────────────────────── */
  confSlider.addEventListener('input', () => {
    const v = confSlider.value / 100;
    confValEl.textContent = v.toFixed(2);
    if (currentDets.length > 0 && lastBitmap) redrawAll(currentDets, true);
  });

  /* ── 초기화 ──────────────────────────────────────────── */
  async function init() {
    // ORT 로드 대기 (CDN fallback 있을 수 있으므로 poll)
    let waited = 0;
    while (typeof ort === 'undefined') {
      await new Promise(r => setTimeout(r, 100));
      waited += 100;
      if (waited > 15000) throw new Error('onnxruntime-web 로드 타임아웃');
    }
    try {
      ort.env.wasm.wasmPaths = window._ortFromCDN
        ? 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/'
        : new URL('./ort/', document.baseURI).href;
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.simd = true;

      const t0 = performance.now();
      const metaRes = await fetch('./metadata.json', { cache: 'no-cache' });
      if (!metaRes.ok) throw new Error(`metadata.json 로드 실패 (${metaRes.status})`);
      META = await metaRes.json();

      SESSION = await ort.InferenceSession.create('./model.onnx', {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      });
      const dt = (performance.now() - t0) / 1000;

      const inName  = SESSION.inputNames[0];
      const outName = SESSION.outputNames[0];
      META._inName  = inName;
      META._outName = outName;

      badgeEl.textContent =
        `${META.model_label}  ·  ${META.img_size}px  ·  ` +
        `${META.class_names.length} class  ·  loaded ${dt.toFixed(1)}s`;
      renderInfo();
      statEl.textContent = '이미지를 선택하면 탐지가 활성화됩니다.';
    } catch (e) {
      console.error(e);
      badgeEl.innerHTML = `<span class="err">모델 로딩 실패: ${e.message}</span>`;
      statEl.innerHTML  = `<span class="err">${e.message}</span>`;
    }
  }

  /* ── 모델 정보 카드 ──────────────────────────────────── */
  function renderInfo() {
    const m = META;
    infoEl.innerHTML = `
      <div class="chips">
        <span class="chip">task: detection (YOLOv8)</span>
        <span class="chip">model: ${m.model_name}</span>
        <span class="chip">input: ${m.img_size}×${m.img_size}</span>
        <span class="chip">classes: ${m.class_names.join(', ')}</span>
        <span class="chip">runtime: onnxruntime-web (wasm)</span>
      </div>
      <div class="stat-grid">
        <div class="stat-box">
          <div class="sv">mAP50 ${(m.full_model_map50 * 100).toFixed(1)}%</div>
          <div class="sk">Full model (${m.full_model_name}, ${m.full_model_size_mb} MB)</div>
        </div>
        <div class="stat-box">
          <div class="sv">mAP50 ${(m.lite_model_map50 * 100).toFixed(1)}%</div>
          <div class="sk">Lite model — 앱 탑재 (${m.lite_model_size_mb} MB)</div>
        </div>
        <div class="stat-box">
          <div class="sv">${m.dataset_total}</div>
          <div class="sk">X-ray 이미지 (train ${m.dataset_train} / val ${m.dataset_val} / test ${m.dataset_test})</div>
        </div>
        <div class="stat-box">
          <div class="sv">Optuna</div>
          <div class="sk">AutoML HPO (TPE + MedianPruner, ${m.source_run})</div>
        </div>
      </div>
    `;
  }

  /* ── 파일 선택 + 드래그 ───────────────────────────────── */
  ['dragenter','dragover'].forEach(ev =>
    dropEl.addEventListener(ev, e => { e.preventDefault(); dropEl.classList.add('dragover'); })
  );
  ['dragleave','drop'].forEach(ev =>
    dropEl.addEventListener(ev, e => { e.preventDefault(); dropEl.classList.remove('dragover'); })
  );
  dropEl.addEventListener('drop', e => {
    const f = e.dataTransfer?.files?.[0]; if (f) handleFile(f);
  });
  fileEl.addEventListener('change', e => {
    const f = e.target.files?.[0]; if (f) handleFile(f);
  });

  async function handleFile(f) {
    if (!f.type.startsWith('image/')) { setError('이미지 파일만 업로드 가능합니다.'); return; }
    lastFileName = f.name;
    statEl.classList.remove('err');
    statEl.textContent = `${f.name} (${(f.size/1024).toFixed(1)} KB)`;
    currentDets = [];

    const url = URL.createObjectURL(f);
    try {
      lastBitmap = await createImageBitmap(f);
    } catch {
      lastBitmap = await new Promise((ok, ng) => {
        const img = new Image();
        img.onload = () => ok(img);
        img.onerror = () => ng(new Error('이미지 디코딩 실패'));
        img.src = url;
      });
    }

    // 이미지 캔버스에 표시
    const W = lastBitmap.naturalWidth || lastBitmap.width;
    const H = lastBitmap.naturalHeight || lastBitmap.height;
    srcCanvas.width = W; srcCanvas.height = H;
    detCanvas.width = W; detCanvas.height = H;
    const ctx = srcCanvas.getContext('2d');
    ctx.drawImage(lastBitmap, 0, 0);
    canvasWrap.classList.add('active');

    resEl.innerHTML = '<em style="color:var(--muted);font-size:12px;">탐지 버튼을 눌러 실행하세요.</em>';
    runEl.disabled = !SESSION;
  }

  /* ── 전처리: letterbox ───────────────────────────────── */
  function letterbox(bitmap, size) {
    const W = bitmap.naturalWidth || bitmap.width;
    const H = bitmap.naturalHeight || bitmap.height;
    const scale = Math.min(size / W, size / H);
    const newW = Math.round(W * scale);
    const newH = Math.round(H * scale);
    const padX = Math.floor((size - newW) / 2);
    const padY = Math.floor((size - newH) / 2);

    const canvas = document.createElement('canvas');
    canvas.width = size; canvas.height = size;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    // 패딩 색 (114, 114, 114)
    ctx.fillStyle = `rgb(114,114,114)`;
    ctx.fillRect(0, 0, size, size);
    ctx.drawImage(bitmap, padX, padY, newW, newH);

    return { canvas, scale, padX, padY, origW: W, origH: H };
  }

  function canvasToFloat32(canvas, size) {
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    const { data } = ctx.getImageData(0, 0, size, size);
    const N = size * size;
    const out = new Float32Array(3 * N);
    for (let i = 0; i < N; i++) {
      out[i]         = data[i*4    ] / 255;  // R
      out[i + N]     = data[i*4 + 1] / 255;  // G
      out[i + 2*N]   = data[i*4 + 2] / 255;  // B
    }
    return out;
  }

  /* ── 후처리 ──────────────────────────────────────────── */
  function parseYoloOutput(rawData, dims, confThresh, scale, padX, padY, origW, origH) {
    // YOLOv8 output: [1, 4+num_cls, num_anchors]
    // dims = [1, numPred, numAnchors]
    const numPred    = dims[1];  // 4 + num_classes
    const numAnchors = dims[2];
    const numCls = numPred - 4;
    const boxes = [];

    for (let ai = 0; ai < numAnchors; ai++) {
      // xc, yc, w, h (absolute pixels in letterboxed image)
      const xc = rawData[0 * numAnchors + ai];
      const yc = rawData[1 * numAnchors + ai];
      const bw = rawData[2 * numAnchors + ai];
      const bh = rawData[3 * numAnchors + ai];

      let maxScore = 0, maxCls = 0;
      for (let c = 0; c < numCls; c++) {
        const s = rawData[(4 + c) * numAnchors + ai];
        if (s > maxScore) { maxScore = s; maxCls = c; }
      }
      if (maxScore < confThresh) continue;

      // 원본 이미지 좌표로 역변환 (letterbox 역산)
      const size = META.img_size;
      const x1 = ((xc - bw / 2) - padX) / scale;
      const y1 = ((yc - bh / 2) - padY) / scale;
      const x2 = ((xc + bw / 2) - padX) / scale;
      const y2 = ((yc + bh / 2) - padY) / scale;
      boxes.push({
        x1: Math.max(0, x1), y1: Math.max(0, y1),
        x2: Math.min(origW, x2), y2: Math.min(origH, y2),
        score: maxScore, cls: maxCls,
      });
    }
    return nmsCpu(boxes, META.iou_threshold || 0.45);
  }

  function iou(a, b) {
    const ix1 = Math.max(a.x1, b.x1), iy1 = Math.max(a.y1, b.y1);
    const ix2 = Math.min(a.x2, b.x2), iy2 = Math.min(a.y2, b.y2);
    const inter = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
    const aA = (a.x2 - a.x1) * (a.y2 - a.y1);
    const bA = (b.x2 - b.x1) * (b.y2 - b.y1);
    return inter / (aA + bA - inter + 1e-6);
  }

  function nmsCpu(boxes, iouThresh) {
    boxes.sort((a, b) => b.score - a.score);
    const keep = [];
    const suppressed = new Uint8Array(boxes.length);
    for (let i = 0; i < boxes.length; i++) {
      if (suppressed[i]) continue;
      keep.push(boxes[i]);
      for (let j = i + 1; j < boxes.length; j++) {
        if (!suppressed[j] && boxes[i].cls === boxes[j].cls && iou(boxes[i], boxes[j]) > iouThresh)
          suppressed[j] = 1;
      }
    }
    return keep;
  }

  /* ── 박스 그리기 ─────────────────────────────────────── */
  function redrawAll(dets, onlyBoxes) {
    const W = srcCanvas.width, H = srcCanvas.height;
    if (!onlyBoxes) {
      const ctx = srcCanvas.getContext('2d');
      ctx.drawImage(lastBitmap, 0, 0, W, H);
    }
    const ctx2 = detCanvas.getContext('2d');
    ctx2.clearRect(0, 0, W, H);
    const confThresh = confSlider.value / 100;
    const visible = dets.filter(d => d.score >= confThresh);

    visible.forEach((det, idx) => {
      const color = BOX_COLORS[idx % BOX_COLORS.length];
      const lw = Math.max(2, Math.round(W / 250));
      const { x1, y1, x2, y2, score } = det;
      const bw = x2 - x1, bh = y2 - y1;
      ctx2.strokeStyle = color;
      ctx2.lineWidth = lw;
      ctx2.strokeRect(x1, y1, bw, bh);

      const label = `fracture ${(score * 100).toFixed(0)}%`;
      const fs = Math.max(11, Math.round(W / 40));
      ctx2.font = `bold ${fs}px sans-serif`;
      const tw = ctx2.measureText(label).width;
      const th = fs + 4;
      const ly = y1 > th + 2 ? y1 - 2 : y1 + bh + th;
      ctx2.fillStyle = color;
      ctx2.fillRect(x1, ly - th, tw + 8, th + 2);
      ctx2.fillStyle = '#000';
      ctx2.fillText(label, x1 + 4, ly - 2);
    });
  }

  /* ── 추론 ────────────────────────────────────────────── */
  async function runDetection() {
    if (!SESSION || !lastBitmap) return;
    runEl.disabled = true;
    statEl.classList.remove('err');
    statEl.innerHTML = '<span class="loader"></span> 추론 중…';
    resEl.innerHTML = '<em style="color:var(--muted);font-size:12px;">처리 중…</em>';

    const size = META.img_size;
    const confThresh = confSlider.value / 100;

    const t0 = performance.now();
    try {
      const { canvas, scale, padX, padY, origW, origH } = letterbox(lastBitmap, size);
      const float32 = canvasToFloat32(canvas, size);
      const tensor = new ort.Tensor('float32', float32, [1, 3, size, size]);
      const inName = META._inName || SESSION.inputNames[0];
      const out = await SESSION.run({ [inName]: tensor });
      const outTensor = out[Object.keys(out)[0]];
      const rawData = outTensor.data;
      const dims = outTensor.dims;  // [1, 5, 8400] for single class

      // dims 확인 및 transpose 필요 여부
      let finalData = rawData, finalDims = dims;
      if (dims.length === 3 && dims[1] < dims[2]) {
        // Already [1, 5, 8400] — correct
      } else if (dims.length === 3 && dims[1] > dims[2]) {
        // [1, 8400, 5] — need to transpose
        const d1 = dims[1], d2 = dims[2];
        const t = new Float32Array(rawData.length);
        for (let i = 0; i < d1; i++)
          for (let j = 0; j < d2; j++)
            t[j * d1 + i] = rawData[i * d2 + j];
        finalData = t;
        finalDims = [1, d2, d1];
      }

      const dt = performance.now() - t0;
      currentDets = parseYoloOutput(finalData, finalDims, confThresh * 0.1, scale, padX, padY, origW, origH);

      redrawAll(currentDets, false);
      renderDetResult(currentDets, confThresh, dt);
      statEl.textContent = `✅ 완료 (${dt.toFixed(0)}ms · ${currentDets.length}개 박스)`;
    } catch (e) {
      console.error(e);
      setError('추론 실패: ' + e.message);
      resEl.innerHTML = `<span class="err">탐지 실패: ${e.message}</span>`;
    } finally {
      runEl.disabled = false;
    }
  }

  /* ── 결과 렌더 ───────────────────────────────────────── */
  function renderDetResult(dets, confThresh, ms) {
    const visible = dets.filter(d => d.score >= confThresh);
    if (dets.length === 0) {
      resEl.innerHTML = `
        <div class="no-det">
          <div class="ico">✅</div>
          <div>골절 탐지 없음</div>
          <div style="margin-top:4px; font-size:10px;">신뢰도 ${(confThresh*100).toFixed(0)}% 이상 박스 없음 · ${ms.toFixed(0)}ms</div>
        </div>`;
      return;
    }

    const topScore = Math.max(...visible.map(d => d.score));
    const color = topScore > 0.7 ? 'var(--bad)' : topScore > 0.4 ? 'var(--warn)' : 'var(--muted)';
    const label = topScore > 0.7 ? '— 높은 신뢰도' : topScore > 0.4 ? '— 중간 신뢰도' : '— 낮은 신뢰도';

    let html = `
      <div class="det-summary">
        <div class="label">🩻 탐지 결과</div>
        <div class="value">${visible.length}개 골절 탐지</div>
        <div class="conf" style="color:${color};">
          최고 신뢰도 ${(topScore*100).toFixed(1)}% ${label} · ${ms.toFixed(0)}ms
        </div>
      </div>
      <div class="det-list">`;
    visible.forEach((det, i) => {
      const color = BOX_COLORS[i % BOX_COLORS.length];
      const w = Math.round(det.x2 - det.x1), h = Math.round(det.y2 - det.y1);
      html += `
        <div class="det-item">
          <div class="dot" style="background:${color};"></div>
          <div class="info">
            <div>fracture <span class="score">${(det.score*100).toFixed(1)}%</span></div>
            <div class="coords">
              x:${Math.round(det.x1)} y:${Math.round(det.y1)} · ${w}×${h}px
            </div>
          </div>
        </div>`;
    });
    html += '</div>';
    if (dets.length > visible.length)
      html += `<div style="margin-top:6px; font-size:10px; color:var(--muted);">+ ${dets.length - visible.length}개 낮은 신뢰도 (슬라이더 조절로 표시)</div>`;
    resEl.innerHTML = html;
  }

  /* ── 헬퍼 ────────────────────────────────────────────── */
  function setError(msg) { statEl.classList.add('err'); statEl.textContent = msg; }

  runEl.addEventListener('click', runDetection);
  confSlider.addEventListener('change', () => {
    if (currentDets.length > 0 && lastBitmap) {
      redrawAll(currentDets, false);
      renderDetResult(currentDets, confSlider.value / 100, 0);
    }
  });
  document.addEventListener('DOMContentLoaded', init);
  if (document.readyState !== 'loading') init();
})();
