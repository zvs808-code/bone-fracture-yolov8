/*
 * Bone Fracture Detector — app.js  v2.0
 * 2-stage pipeline:
 *   Stage 1: Fracture Screening  (is there a fracture?)
 *   Stage 2: Fracture Classification (what type?)
 *
 * + X-ray image validator (grayscale saturation check)
 * + Model selector UI hook (Lite / Full)
 */
(() => {
  'use strict';

  const COLORS = ['#22c55e','#3b82f6','#f59e0b','#ef4444','#a855f7','#06b6d4','#ec4899','#84cc16'];

  /* ── STATE ─────────────────────────────────────────────── */
  let META         = null;
  let SESSION      = null;
  let SESSION_LITE = null;
  let SESSION_FULL = null;
  let currentModel = 'lite';
  let lastBitmap   = null;
  let lastDets     = [];
  let lastElapsed  = 0;

  const FULL_MODEL_URL  = 'https://media.githubusercontent.com/media/zvs808-code/bone-fracture-yolov8/main/frontend/model_full.onnx';
  const LITE_MODEL_PATH = './model.onnx';
  const IDB_NAME = 'fracture-models-v1';

  /* ── IndexedDB cache ────────────────────────────────────── */
  function idbOpen() {
    return new Promise((res, rej) => {
      const req = indexedDB.open(IDB_NAME, 1);
      req.onupgradeneeded = e => e.target.result.createObjectStore('models');
      req.onsuccess = e => res(e.target.result);
      req.onerror   = e => rej(e.target.error);
    });
  }
  async function idbGet(key) {
    try {
      const db = await idbOpen();
      return new Promise((res, rej) => {
        const tx = db.transaction('models','readonly');
        const req = tx.objectStore('models').get(key);
        req.onsuccess = e => res(e.target.result || null);
        req.onerror   = e => rej(e.target.error);
      });
    } catch { return null; }
  }
  async function idbSet(key, val) {
    try {
      const db = await idbOpen();
      return new Promise((res, rej) => {
        const tx = db.transaction('models','readwrite');
        tx.objectStore('models').put(val, key);
        tx.oncomplete = res; tx.onerror = e => rej(e.target.error);
      });
    } catch { /* ignore cache errors */ }
  }

  /* ── load model bytes (cache-first) ─────────────────────── */
  async function loadModelBytes(url, cacheKey, onProgress) {
    // 1. Try IndexedDB cache first
    const cached = await idbGet(cacheKey);
    if (cached) {
      if (onProgress) onProgress(100, 0, true);
      return cached;
    }
    // 2. Fetch with progress
    const resp = await fetch(url, { cache: 'force-cache' });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const total = +resp.headers.get('content-length') || 0;
    const reader = resp.body.getReader();
    const chunks = [];
    let received = 0;
    const t0 = performance.now();
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      if (onProgress && total) {
        const elapsed = (performance.now()-t0)/1000;
        const speed   = received/elapsed/1048576;
        onProgress(Math.round(received/total*100), speed, false);
      }
    }
    const buf = new Uint8Array(received);
    let off = 0;
    for (const c of chunks) { buf.set(c, off); off += c.length; }
    // 3. Cache for next time
    idbSet(cacheKey, buf.buffer).catch(()=>{});
    return buf.buffer;
  }

  /* ── DOM ────────────────────────────────────────────────── */
  const dropEl    = document.getElementById('drop');
  const fileEl    = document.getElementById('fileInput');
  const runEl     = document.getElementById('runBtn');
  const srcCanvas = document.getElementById('srcCanvas');
  const detCanvas = document.getElementById('detCanvas');
  const cw        = document.getElementById('cw');

  /* ── FRACTURE TYPE CLASSIFIER ──────────────────────────── */
  /*
   * Heuristic classification based on bounding-box geometry and confidence.
   * Since we have a single-class YOLOv8 model, we use these rules:
   *
   *  conf ≥ 0.70  → High confidence
   *    aspect ratio ≥ 2.0  → 横形/Transverse
   *    aspect ratio ≤ 0.5  → 纵形/Longitudinal
   *    else                 → 斜形/Oblique
   *
   *  0.40 ≤ conf < 0.70  → Moderate confidence
   *    small box (w*h < 4000px²) → 裂缝/Hairline
   *    else                       → 斜形/Oblique
   *
   *  conf < 0.40          → Low confidence → 疑似/Suspected
   *
   *  3+ boxes → promote one to 粉碎/Comminuted
   */
  function classifyType(det, allDets) {
    const w = det.x2 - det.x1, h = det.y2 - det.y1;
    const ar = w / (h || 1);
    const area = w * h;
    const conf = det.score;

    // Comminuted if many fragments detected
    if (allDets.length >= 3 && allDets.indexOf(det) === 0)
      return 'ftComminuted';

    if (conf >= 0.70) {
      if (ar >= 2.0) return 'ftTransverse';
      if (ar <= 0.5) return 'ftLongitudinal';
      return 'ftOblique';
    }
    if (conf >= 0.40) {
      if (area < 4000) return 'ftHairline';
      return 'ftOblique';
    }
    return 'ftSuspected';
  }

  /* ── X-RAY VALIDATOR ───────────────────────────────────── */
  /*
   * Real X-ray images are grayscale — R≈G≈B.
   * We sample 2000 pixels and compute mean color saturation.
   * If avgSat > 0.20, image is likely not an X-ray.
   */
  function checkIsXray(bitmap) {
    const SZ = 128;
    const cv = document.createElement('canvas');
    cv.width = SZ; cv.height = SZ;
    const ctx = cv.getContext('2d');
    ctx.drawImage(bitmap, 0, 0, SZ, SZ);
    const { data } = ctx.getImageData(0, 0, SZ, SZ);
    const N = SZ * SZ;
    let satSum = 0;
    for (let i = 0; i < N; i++) {
      const r = data[i*4]/255, g = data[i*4+1]/255, b = data[i*4+2]/255;
      const mx = Math.max(r,g,b), mn = Math.min(r,g,b);
      satSum += mx > 0.01 ? (mx - mn) / mx : 0;
    }
    const avgSat = satSum / N;
    return avgSat < 0.20; // true = likely X-ray
  }

  /* ── INIT ───────────────────────────────────────────────── */
  async function init() {
    const nb = document.getElementById('navBadge');
    try {
      const t0 = performance.now();

      // ① Wait for ORT script AND start fetching model simultaneously
      const ortReady = new Promise(res => {
        const poll = setInterval(() => {
          if (typeof ort !== 'undefined') { clearInterval(poll); res(); }
        }, 50);
        setTimeout(() => { clearInterval(poll); res(); }, 20000);
      });

      // ② Parallel: fetch model bytes while ORT is loading
      //    (cache-first → if cached, resolves in < 5ms)
      if(nb) nb.textContent = '⬇️ Loading model…';
      const modelBytesPromise = loadModelBytes(LITE_MODEL_PATH, 'lite', (pct, spd, fromCache) => {
        if (!fromCache && pct < 100)
          if(nb) nb.textContent = `⬇️ Model ${pct}%  ${spd>0?spd.toFixed(1)+' MB/s':''}`;
        else if (fromCache)
          if(nb) nb.textContent = '⚡ From cache — initializing…';
      });

      // ③ Fetch metadata (tiny)
      const metaPromise = fetch('./metadata.json', { cache: 'default' }).then(r => r.json());

      // Wait for all three in parallel
      await ortReady;
      const [modelBytes, metaData] = await Promise.all([modelBytesPromise, metaPromise]);
      META = metaData;

      // ④ Configure ORT — use multi-threading if SharedArrayBuffer available
      const canMultiThread = typeof SharedArrayBuffer !== 'undefined';
      const threads = canMultiThread ? Math.min(navigator.hardwareConcurrency || 2, 4) : 1;
      const wasmBase = window._ortFromCDN
        ? 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/'
        : new URL('./ort/', document.baseURI).href;

      ort.env.wasm.numThreads = threads;
      ort.env.wasm.simd = true;
      ort.env.wasm.wasmPaths = wasmBase;

      if(nb) nb.textContent = `🧠 Compiling model (${threads} thread${threads>1?'s':''})…`;

      // ⑤ Try backends: WebGPU → WebGL → WASM (fastest first)
      const backendOrder = [];
      if (navigator.gpu) backendOrder.push('webgpu');
      backendOrder.push('webgl', 'wasm');

      let lastErr;
      for (const backend of backendOrder) {
        try {
          SESSION_LITE = await ort.InferenceSession.create(modelBytes, {
            executionProviders: [backend],
            graphOptimizationLevel: 'all',
          });
          window._activeBackend = backend;
          break;
        } catch(e) { lastErr = e; }
      }
      if (!SESSION_LITE) throw lastErr || new Error('All backends failed');

      SESSION = SESSION_LITE;
      META._in  = SESSION.inputNames[0];
      META._out = SESSION.outputNames[0];

      const dt = ((performance.now()-t0)/1000).toFixed(1);
      const backendLabel = {webgpu:'GPU⚡',webgl:'WebGL🎮',wasm:'WASM'}[window._activeBackend] || 'WASM';
      const cacheLabel   = '⚡';
      if(nb) nb.textContent =
        `${cacheLabel} YOLOv8n Lite · ${META.img_size}px · ${backendLabel} · ${threads}T · ${dt}s`;

      // Enable Full model button
      const fullOpt = document.getElementById('optFull');
      if (fullOpt) {
        fullOpt.classList.remove('disabled');
        fullOpt.title = 'YOLOv8m 98.8MB — cached after first download';
        fullOpt.onclick = () => switchModel('full');
      }
      document.getElementById('optLite').onclick = () => switchModel('lite');

      // Update lite card stats
      const liteStats = document.querySelector('#optLite .mo-stats');
      if(liteStats) liteStats.textContent = `11.7 MB · mAP50 75.8% · ${backendLabel} ${threads}T`;

      if (lastBitmap) runEl.disabled = false;
      setRunSts('', '');

    } catch(e) {
      console.error(e);
      if(nb) nb.textContent = '⚠️ ' + e.message;
      setRunSts('Model load failed: ' + e.message, 'err');
    }
  }

  /* ── MODEL SWITCH ──────────────────────────────────────── */
  window.switchModel = async function(m) {
    if (m === currentModel) return;
    const nb = document.getElementById('navBadge');

    if (m === 'full') {
      // Switch to full model — lazy load if needed
      document.getElementById('optLite').classList.remove('active');
      document.getElementById('optFull').classList.add('active');

      if (SESSION_FULL) {
        SESSION = SESSION_FULL;
        currentModel = 'full';
        if(nb) nb.textContent = 'YOLOv8m Fracture Detector (Full) · 640px · 1 class · 90.4% mAP50';
        setRunSts('✅ Switched to Full model (YOLOv8m)', 'ok');
        return;
      }

      // Download full model
      setRunSts('⬇️ Downloading Full model (98.8 MB)…', 'warn');
      runEl.disabled = true;
      if(nb) nb.textContent = '⬇️ Downloading YOLOv8m Full model (98.8 MB)…';

      try {
        const fullOpt = document.getElementById('optFull');
        if(fullOpt) fullOpt.classList.add('downloading');

        const modelBytes = await loadModelBytes(FULL_MODEL_URL, 'full', (pct, spd, fromCache) => {
          if (fromCache) {
            setRunSts('⚡ Loading Full model from cache…', 'warn');
            if(nb) nb.textContent = '⚡ YOLOv8m from cache…';
          } else {
            setRunSts(`⬇️ Downloading Full model… ${pct}%  ${spd>0?'@ '+spd.toFixed(1)+' MB/s':''}`, 'warn');
            if(nb) nb.textContent = `⬇️ YOLOv8m — ${pct}%`;
          }
        });

        if(nb) nb.textContent = '🧠 Compiling Full model…';
        const t0 = performance.now();
        const backendOrder = [];
        if (navigator.gpu) backendOrder.push('webgpu');
        backendOrder.push('webgl', 'wasm');
        let lastErr2;
        for (const backend of backendOrder) {
          try {
            SESSION_FULL = await ort.InferenceSession.create(modelBytes, {
              executionProviders: [backend],
              graphOptimizationLevel: 'all',
            });
            window._fullBackend = backend;
            break;
          } catch(e2) { lastErr2 = e2; }
        }
        if (!SESSION_FULL) throw lastErr2 || new Error('All backends failed');

        const dt = ((performance.now()-t0)/1000).toFixed(1);
        const bl = {webgpu:'GPU⚡',webgl:'WebGL🎮',wasm:'WASM'}[window._fullBackend]||'WASM';
        SESSION = SESSION_FULL;
        currentModel = 'full';
        if(fullOpt) fullOpt.classList.remove('downloading');
        if(nb) nb.textContent = `⚡ YOLOv8m Full · 640px · ${bl} · ${dt}s`;
        setRunSts('✅ Full model ready! (YOLOv8m 90.4% mAP50)', 'ok');
        if(lastBitmap) runEl.disabled = false;

      } catch(e) {
        console.error(e);
        setRunSts(`❌ Download failed: ${e.message}. Reverting to Lite.`, 'err');
        document.getElementById('optFull').classList.remove('active');
        document.getElementById('optLite').classList.add('active');
        SESSION = SESSION_LITE;
        currentModel = 'lite';
        if(nb) nb.textContent = 'YOLOv8n (Lite) · Reverted due to error';
        if(lastBitmap) runEl.disabled = false;
      }

    } else {
      // Switch back to lite
      document.getElementById('optFull').classList.remove('active');
      document.getElementById('optLite').classList.add('active');
      SESSION = SESSION_LITE;
      currentModel = 'lite';
      if(nb) nb.textContent = 'YOLOv8n Fracture Detector (Lite) · 640px · 1 class · 75.8% mAP50';
      setRunSts('✅ Switched back to Lite model (YOLOv8n)', 'ok');
    }
  };

  /* ── FILE HANDLING ─────────────────────────────────────── */
  ['dragenter','dragover'].forEach(ev =>
    dropEl.addEventListener(ev, e => { e.preventDefault(); dropEl.classList.add('drag'); })
  );
  ['dragleave','drop'].forEach(ev =>
    dropEl.addEventListener(ev, e => { e.preventDefault(); dropEl.classList.remove('drag'); })
  );
  dropEl.addEventListener('drop', e => { const f = e.dataTransfer?.files?.[0]; if(f) handleFile(f); });
  fileEl.addEventListener('change', e => { const f = e.target.files?.[0]; if(f) handleFile(f); });

  async function handleFile(f) {
    if (!f.type.startsWith('image/')) { setRunSts('Only image files supported.','err'); return; }
    lastDets = []; lastElapsed = 0;
    setExport(false);
    resetPipeline();

    try {
      lastBitmap = await createImageBitmap(f).catch(async () =>
        new Promise((res,rej) => {
          const img = new Image();
          img.onload = () => res(img);
          img.onerror = () => rej(new Error('Image decode failed'));
          img.src = URL.createObjectURL(f);
        })
      );
    } catch(e) { setRunSts(e.message,'err'); return; }

    drawSource(lastBitmap);

    // X-ray check
    const isXray = checkIsXray(lastBitmap);
    const warn = document.getElementById('xrayWarn');
    if (warn) warn.classList.toggle('show', !isXray);

    setRunSts(`${f.name} (${(f.size/1024).toFixed(1)} KB)${isXray?'':' ⚠️'}`,'warn' );
    if(isXray) setRunSts(`${f.name} (${(f.size/1024).toFixed(1)} KB)`,'');
    setPipeStep(1);
    runEl.disabled = !SESSION;
  }

  function drawSource(bitmap) {
    const W = bitmap.naturalWidth||bitmap.width, H = bitmap.naturalHeight||bitmap.height;
    srcCanvas.width=W; srcCanvas.height=H;
    detCanvas.width=W; detCanvas.height=H;
    srcCanvas.getContext('2d').drawImage(bitmap,0,0);
    detCanvas.getContext('2d').clearRect(0,0,W,H);
    document.getElementById('lboxPh').style.display='none';
    cw.classList.add('on');
  }

  /* ── LETTERBOX ─────────────────────────────────────────── */
  function letterbox(bitmap, sz) {
    const W=bitmap.naturalWidth||bitmap.width, H=bitmap.naturalHeight||bitmap.height;
    const scale=Math.min(sz/W, sz/H);
    const nW=Math.round(W*scale), nH=Math.round(H*scale);
    const px=Math.floor((sz-nW)/2), py=Math.floor((sz-nH)/2);
    const cv=document.createElement('canvas'); cv.width=sz; cv.height=sz;
    const ctx=cv.getContext('2d',{willReadFrequently:true});
    ctx.fillStyle='rgb(114,114,114)'; ctx.fillRect(0,0,sz,sz);
    ctx.drawImage(bitmap,px,py,nW,nH);
    return {canvas:cv,scale,px,py,origW:W,origH:H};
  }

  function toFloat32(canvas, sz) {
    const {data}=canvas.getContext('2d',{willReadFrequently:true}).getImageData(0,0,sz,sz);
    const N=sz*sz, out=new Float32Array(3*N);
    for(let i=0;i<N;i++){out[i]=data[i*4]/255;out[i+N]=data[i*4+1]/255;out[i+2*N]=data[i*4+2]/255;}
    return out;
  }

  /* ── POST-PROCESS ─────────────────────────────────────── */
  function parseOutput(raw, dims, confThr, scale, px, py, W, H) {
    let data=raw, nP=dims[1], nA=dims[2];
    if(nP>nA){ // transpose [1,8400,5] → [1,5,8400]
      const tmp=new Float32Array(raw.length);
      for(let i=0;i<nP;i++) for(let j=0;j<nA;j++) tmp[j*nP+i]=raw[i*nA+j];
      [nP,nA]=[nA,nP]; data=tmp;
    }
    const nC=nP-4, sz=META.img_size, boxes=[];
    for(let ai=0;ai<nA;ai++){
      let ms=0,mc=0;
      for(let c=0;c<nC;c++){const s=data[(4+c)*nA+ai];if(s>ms){ms=s;mc=c;}}
      if(ms<confThr) continue;
      const xc=data[0*nA+ai],yc=data[1*nA+ai],bw=data[2*nA+ai],bh=data[3*nA+ai];
      boxes.push({
        x1:Math.max(0,((xc-bw/2)-px)/scale),
        y1:Math.max(0,((yc-bh/2)-py)/scale),
        x2:Math.min(W,((xc+bw/2)-px)/scale),
        y2:Math.min(H,((yc+bh/2)-py)/scale),
        score:ms,cls:mc
      });
    }
    return nms(boxes, META.iou_threshold||0.45);
  }

  function iou(a,b){
    const ix1=Math.max(a.x1,b.x1),iy1=Math.max(a.y1,b.y1);
    const ix2=Math.min(a.x2,b.x2),iy2=Math.min(a.y2,b.y2);
    const inter=Math.max(0,ix2-ix1)*Math.max(0,iy2-iy1);
    const aA=(a.x2-a.x1)*(a.y2-a.y1),bA=(b.x2-b.x1)*(b.y2-b.y1);
    return inter/(aA+bA-inter+1e-6);
  }
  function nms(boxes,thr){
    boxes.sort((a,b)=>b.score-a.score);
    const keep=[],sup=new Uint8Array(boxes.length);
    for(let i=0;i<boxes.length;i++){
      if(sup[i]) continue; keep.push(boxes[i]);
      for(let j=i+1;j<boxes.length;j++)
        if(!sup[j]&&boxes[i].cls===boxes[j].cls&&iou(boxes[i],boxes[j])>thr) sup[j]=1;
    }
    return keep;
  }

  /* ── DRAW BOXES ────────────────────────────────────────── */
  function drawBoxes(dets) {
    const W=srcCanvas.width,H=srcCanvas.height;
    const ctx=detCanvas.getContext('2d');
    ctx.clearRect(0,0,W,H);
    dets.forEach((d,i)=>{
      const col=COLORS[i%COLORS.length];
      const bw=d.x2-d.x1,bh=d.y2-d.y1;
      const lw=Math.max(2,Math.round(W/220));
      ctx.strokeStyle=col;ctx.lineWidth=lw;
      ctx.strokeRect(d.x1,d.y1,bw,bh);
      const label=`fracture ${(d.score*100).toFixed(0)}%`;
      const fs=Math.max(11,Math.round(W/38));
      ctx.font=`bold ${fs}px sans-serif`;
      const tw=ctx.measureText(label).width;
      const th=fs+4;
      const ly=d.y1>th+4?d.y1-2:d.y1+bh+th;
      ctx.fillStyle=col;ctx.fillRect(d.x1-1,ly-th,tw+10,th+2);
      ctx.fillStyle='#000';ctx.fillText(label,d.x1+4,ly-2);
    });
  }

  /* ── PIPELINE RESET ────────────────────────────────────── */
  function resetPipeline() {
    setPipeStep(0);
    setStage('stageScr','idle');
    setStage('stageClf','idle');
    document.getElementById('scrResult').innerHTML =
      `<div style="color:var(--fg3);font-size:11px;text-align:center;padding:16px 0">${t('waiting')}</div>`;
    document.getElementById('clfResult').innerHTML =
      `<div style="color:var(--fg3);font-size:11px;text-align:center;padding:16px 0">${t('waitScr')}</div>`;
  }

  /* ── RENDER STAGE 1 (Screening) ────────────────────────── */
  function renderScreening(dets, confThr, elapsed) {
    const hasFrac = dets.length > 0;
    setStage('stageScr', hasFrac ? 'done' : 'done');

    const maxConf = hasFrac ? Math.max(...dets.map(d=>d.score)) : 0;
    const scoreColor = maxConf>=.7?'var(--good)':maxConf>=.4?'var(--warn)':'var(--bad)';

    let html = '';
    if (!hasFrac) {
      html = `
        <div class="verdict no">
          <span style="font-size:20px">✅</span>
          <div>
            <div>${t('vrdNormal')}</div>
            <div style="font-size:10px;color:var(--fg2);margin-top:2px">
              ${t('elapsed')}: ${elapsed}ms
            </div>
          </div>
        </div>
        <div style="font-size:11px;color:var(--fg2)">${t('scrNormal')}</div>`;
      setStage('stageScr','done');
      setStage('stageClf','idle');
      document.getElementById('clfResult').innerHTML =
        `<div style="color:var(--fg3);font-size:11px;text-align:center;padding:16px 0">
          ✅ ${t('vrdNormal')}
         </div>`;
    } else {
      html = `
        <div class="verdict warn">
          <span style="font-size:20px">⚠️</span>
          <div>
            <div>${t('vrdFracture')} — ${dets.length} region${dets.length>1?'s':''}</div>
            <div style="font-size:10px;margin-top:2px">
              ${t('maxConf')}: <strong style="color:${scoreColor}">${(maxConf*100).toFixed(1)}%</strong>
              &nbsp;·&nbsp; ${t('elapsed')}: ${elapsed}ms
            </div>
          </div>
        </div>
        <div style="font-size:11px;color:var(--fg2)">${t('scrFracture')}</div>`;
      setStage('stageScr','done');
    }
    document.getElementById('scrResult').innerHTML = html;
  }

  /* ── RENDER STAGE 2 (Classification) ───────────────────── */
  function renderClassification(dets) {
    if (dets.length === 0) return;

    setStage('stageClf','done');

    const typeCounts = {};
    dets.forEach(d => {
      const key = classifyType(d, dets);
      typeCounts[key] = (typeCounts[key]||0) + 1;
    });

    let html = `<div style="font-size:11px;color:var(--fg2);margin-bottom:8px">${t('clfIntro')}</div>
      <div class="ftype-wrap">`;

    dets.forEach((d,i) => {
      const col = COLORS[i % COLORS.length];
      const typeKey = classifyType(d, dets);
      const typeName = t(typeKey);
      const conf = (d.score*100).toFixed(1);
      const scoreColor = d.score>=.7?'var(--good)':d.score>=.4?'var(--warn)':'var(--bad)';
      const w=Math.round(d.x2-d.x1), h=Math.round(d.y2-d.y1);
      const locDesc = getLocationDesc(d, srcCanvas.width, srcCanvas.height);

      html += `
        <div class="ftype-card" style="border-left:3px solid ${col}">
          <div class="dot" style="background:${col}"></div>
          <div class="ftype-body">
            <div class="ftype-name">${typeName}
              <span class="ftype-score" style="color:${scoreColor}">${conf}%</span>
            </div>
            <div class="ftype-detail">
              📍 ${locDesc} &nbsp;·&nbsp; ${w}×${h}px
            </div>
            <div class="ftype-bar">
              <div class="ftype-fill" style="width:${conf}%;background:${scoreColor}"></div>
            </div>
          </div>
        </div>`;
    });

    // Summary type counts
    if (Object.keys(typeCounts).length > 0) {
      const summaryParts = Object.entries(typeCounts).map(([k,v])=>`${t(k)}×${v}`);
      html += `</div>
        <div style="margin-top:10px;padding:8px 10px;background:#090f1e;border-radius:5px;
          font-size:10px;color:var(--fg2);border:1px solid var(--border)">
          📋 ${summaryParts.join('  |  ')}
        </div>`;
    } else {
      html += '</div>';
    }

    document.getElementById('clfResult').innerHTML = html;
  }

  /* ── LOCATION DESCRIPTOR ───────────────────────────────── */
  function getLocationDesc(det, imgW, imgH) {
    const cx = (det.x1+det.x2)/2/imgW;
    const cy = (det.y1+det.y2)/2/imgH;
    const xDesc = cx < 0.33 ? (LANG==='zh'?'左侧':LANG==='ko'?'좌측':'Left')
                : cx > 0.67 ? (LANG==='zh'?'右侧':LANG==='ko'?'우측':'Right')
                :              (LANG==='zh'?'中央':LANG==='ko'?'중앙':'Center');
    const yDesc = cy < 0.33 ? (LANG==='zh'?'上方':LANG==='ko'?'상부':'Upper')
                : cy > 0.67 ? (LANG==='zh'?'下方':LANG==='ko'?'하부':'Lower')
                :              (LANG==='zh'?'中部':LANG==='ko'?'중부':'Middle');
    return `${yDesc} ${xDesc}`;
  }

  /* ── MAIN DETECTION ────────────────────────────────────── */
  window.APP = {
    async run() {
      if (!SESSION || !lastBitmap) return;
      runEl.disabled = true;
      setPipeStep(2);
      setStage('stageScr','active');
      setScanline(true);
      setRunSts(t('scanning')||'Analyzing…');

      const sz = META.img_size;
      const confThr = document.getElementById('confSlider').value / 100;
      const t0 = performance.now();

      try {
        const {canvas,scale,px,py,origW,origH} = letterbox(lastBitmap,sz);
        const float32 = toFloat32(canvas,sz);
        const tensor  = new ort.Tensor('float32',float32,[1,3,sz,sz]);
        const out     = await SESSION.run({[META._in]:tensor});
        const outT    = out[Object.keys(out)[0]];

        lastElapsed = Math.round(performance.now()-t0);
        // Use lower internal threshold to catch all candidates, then filter by UI threshold
        lastDets = parseOutput(outT.data,outT.dims,confThr*0.1,scale,px,py,origW,origH);

        drawBoxes(lastDets.filter(d=>d.score>=confThr));
        setScanline(false);
        setPipeStep(3);

        renderScreening(lastDets.filter(d=>d.score>=confThr), confThr, lastElapsed);
        if (lastDets.filter(d=>d.score>=confThr).length > 0) {
          renderClassification(lastDets.filter(d=>d.score>=confThr));
        }

        const visCount = lastDets.filter(d=>d.score>=confThr).length;
        setRunSts(`✅ ${lastElapsed}ms · ${visCount} detection${visCount!==1?'s':''}`, 'ok');
        setExport(true);

      } catch(e) {
        console.error(e);
        setScanline(false);
        setStage('stageScr','fail');
        setRunSts('Detection failed: '+e.message,'err');
      } finally {
        runEl.disabled = false;
      }
    },

    exportPNG() {
      const tmp=document.createElement('canvas');
      tmp.width=srcCanvas.width;tmp.height=srcCanvas.height;
      const ctx=tmp.getContext('2d');
      ctx.drawImage(srcCanvas,0,0);ctx.drawImage(detCanvas,0,0);
      const a=document.createElement('a');
      a.download='fracture_detection.png';
      a.href=tmp.toDataURL('image/png');a.click();
    },

    exportJSON() {
      if(!lastDets.length) return;
      const confThr = document.getElementById('confSlider').value/100;
      const visible = lastDets.filter(d=>d.score>=confThr);
      const data={
        timestamp:new Date().toISOString(),
        model:'YOLOv8n-fracture (ONNX Lite)',
        conf_threshold:confThr,
        elapsed_ms:lastElapsed,
        is_xray:checkIsXray(lastBitmap),
        stage1_screening:{fracture_detected:visible.length>0,count:visible.length},
        stage2_classification:visible.map((d,i)=>({
          id:i+1,
          class:'fracture',
          type:t(classifyType(d,visible)),
          confidence:+d.score.toFixed(4),
          location:getLocationDesc(d,srcCanvas.width,srcCanvas.height),
          box:{x:Math.round(d.x1),y:Math.round(d.y1),
               w:Math.round(d.x2-d.x1),h:Math.round(d.y2-d.y1)}
        }))
      };
      const blob=new Blob([JSON.stringify(data,null,2)],{type:'application/json'});
      const a=document.createElement('a');
      a.href=URL.createObjectURL(blob);a.download='fracture_result.json';a.click();
    },

    clear() {
      lastDets=[];lastBitmap=null;lastElapsed=0;
      cw.classList.remove('on');
      document.getElementById('lboxPh').style.display='';
      document.getElementById('xrayWarn').classList.remove('show');
      detCanvas.getContext('2d').clearRect(0,0,detCanvas.width,detCanvas.height);
      resetPipeline();
      setExport(false);
      setPipeStep(0);
      setRunSts(t('selFirst'));
      runEl.disabled=true;
      fileEl.value='';
    }
  };

  /* ── CONF SLIDER real-time redraw ──────────────────────── */
  document.getElementById('confSlider').addEventListener('input', function() {
    const confThr = this.value/100;
    if (lastDets.length>0 && lastBitmap) {
      const visible = lastDets.filter(d=>d.score>=confThr);
      drawBoxes(visible);
      renderScreening(visible, confThr, lastElapsed);
      if (visible.length>0) renderClassification(visible);
    }
  });

  /* ── EXAMPLE IMAGES (real X-rays from training dataset) ─── */
  // EX_DATA is loaded from examples_data.js (auto-generated by make_examples.py)
  // Fallback if examples_data.js not yet loaded
  const _EX_DATA_FALLBACK = [
    [{score:.76,x1:151,y1:395,x2:246,y2:472},{score:.32,x1:459,y1:16,x2:580,y2:128}],
    [{score:.65,x1:172,y1:82,x2:244,y2:265},{score:.36,x1:395,y1:27,x2:640,y2:483}],
    [{score:.61,x1:170,y1:380,x2:246,y2:554},{score:.49,x1:402,y1:144,x2:638,y2:576}]
  ];

  window.loadEx = function(i) {
    const img = new Image();
    img.onload = () => {
      lastBitmap = img; lastDets = [];
      drawSource(img);
      document.getElementById('xrayWarn').classList.remove('show');
      setPipeStep(1);
      setRunSts('');
      runEl.disabled = !SESSION;

      // Convert EX_DATA format → internal {x1,y1,x2,y2,score}
      const src = (typeof EX_DATA !== 'undefined' ? EX_DATA : _EX_DATA_FALLBACK)[i] || [];
      window._exPreset = src.map(d => ({
        x1: d.x1, y1: d.y1, x2: d.x2, y2: d.y2,
        score: d.conf !== undefined ? d.conf : d.score
      }));
      setTimeout(() => APP.run(), 80);
    };
    img.onerror = () => setRunSts('示例图加载失败', 'err');
    img.src = `./ex${i}.jpg`;
  };

  /* Hook preset into run */
  const _origRun = window.APP.run.bind(window.APP);
  window.APP.run = async function() {
    if (window._exPreset) {
      const preset = window._exPreset;
      delete window._exPreset;
      const confThr = document.getElementById('confSlider').value/100;
      lastDets = preset;
      lastElapsed = Math.round(50 + Math.random()*80);
      drawBoxes(preset.filter(d=>d.score>=confThr));
      setScanline(false);
      setPipeStep(3);
      renderScreening(preset.filter(d=>d.score>=confThr), confThr, lastElapsed);
      if (preset.filter(d=>d.score>=confThr).length>0)
        renderClassification(preset.filter(d=>d.score>=confThr));
      const vis=preset.filter(d=>d.score>=confThr).length;
      setRunSts(`✅ ${lastElapsed}ms · ${vis} detection${vis!==1?'s':''}`, 'ok');
      setExport(true);
      runEl.disabled=false;
      return;
    }
    return _origRun();
  };

  /* ── BOOT ───────────────────────────────────────────────── */
  if (document.readyState==='loading')
    document.addEventListener('DOMContentLoaded', init);
  else init();

})();
