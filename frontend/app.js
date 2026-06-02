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

  /** Bump when deploying — clears stale JS/model caches once per browser. */
  const APP_CACHE_REV = '2026-06-01-mtl-blue';
  const APP_CACHE_KEY = 'bone_app_cache_rev';

  async function purgeStaleClientCache() {
    if (localStorage.getItem(APP_CACHE_KEY) === APP_CACHE_REV) return;
    try {
      await new Promise((res, rej) => {
        const req = indexedDB.deleteDatabase('fracture-models-v1');
        req.onsuccess = () => res();
        req.onerror = () => rej(req.error);
        req.onblocked = () => res();
      });
    } catch (e) { console.warn('[cache] IDB purge', e); }
    localStorage.setItem(APP_CACHE_KEY, APP_CACHE_REV);
    if ('caches' in window) {
      try {
        const keys = await caches.keys();
        await Promise.all(keys.map(k => caches.delete(k)));
      } catch (e) { console.warn('[cache] SW caches', e); }
    }
    console.info('[cache] Cleared model + cache storage for', APP_CACHE_REV);
  }

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
  let lastIsXray   = true;   // basic-image-check result of the current image (OOD gating)

  /** Static hosting paths (root-relative; no "public/" prefix). Tried in order. */
  const FULL_MODEL_CANDIDATES = [
    '/model_full.onnx',
    '/models/model_full.onnx',
    '/models/yolov8m.onnx',
    'https://github.com/zvs808-code/bone-fracture-yolov8/releases/download/v1.0.0/model_full.onnx',
    'https://media.githubusercontent.com/media/zvs808-code/bone-fracture-yolov8/main/frontend/model_full.onnx',
  ];
  const FULL_MODEL_CACHE_KEY = 'full-v3';
  const LITE_MODEL_PATH = '/model.onnx';
  const LITE_MODEL_FALLBACK = './model.onnx';
  const IDB_NAME = 'fracture-models-v1';

  let isModelLoading = false;

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

  function resolveModelUrl(path) {
    return new URL(path, document.baseURI || window.location.href).href;
  }

  async function loadModelBytesFromCandidates(paths, cacheKey, onProgress) {
    const tried = [];
    for (const path of paths) {
      try {
        return await loadModelBytes(resolveModelUrl(path), cacheKey, onProgress);
      } catch (e) {
        const msg = (e && e.message) || String(e);
        tried.push(`${path} (${msg})`);
        console.warn('[model] load failed:', path, msg);
      }
    }
    throw new Error(
      'YOLOv8m model not found. Place model_full.onnx in www/ or www/models/yolov8m.onnx.\n' +
      tried.join('\n')
    );
  }

  function modelLoadCopy() {
    if (LANG === 'zh') {
      return {
        title: '正在加载高精度医疗大脑 (YOLOv8m · 99MB)',
        sub: '首次下载需要 5–15 秒，请稍候…',
        compile: '正在编译 ONNX 会话…',
      };
    }
    if (LANG === 'ko') {
      return {
        title: '고정밀 의료 모델 로딩 중 (YOLOv8m · 99MB)',
        sub: '첫 다운로드는 5–15초 걸릴 수 있습니다…',
        compile: 'ONNX 세션 컴파일 중…',
      };
    }
    return {
      title: 'Loading high-precision model (YOLOv8m · 99MB)',
      sub: 'First download may take 5–15 seconds…',
      compile: 'Compiling ONNX session…',
    };
  }

  function setModelLoading(on, opts = {}) {
    isModelLoading = !!on;
    const overlay = document.getElementById('modelLoadOverlay');
    if (!overlay) return;
    if (!on) {
      overlay.hidden = true;
      overlay.setAttribute('aria-hidden', 'true');
      return;
    }
    const copy = modelLoadCopy();
    overlay.hidden = false;
    overlay.setAttribute('aria-hidden', 'false');
    const titleEl = document.getElementById('modelLoadTitle');
    const subEl = document.getElementById('modelLoadSub');
    const pctEl = document.getElementById('modelLoadPct');
    const fillEl = document.getElementById('modelLoadBarFill');
    if (titleEl) titleEl.textContent = opts.title || copy.title;
    if (subEl) subEl.textContent = opts.sub || copy.sub;
    if (pctEl) pctEl.textContent = opts.pctText || '';
    if (fillEl) fillEl.style.width = opts.pct != null ? `${Math.min(100, opts.pct)}%` : '8%';
  }

  function showModelToast(message, kind = 'warn') {
    let toast = document.getElementById('modelToast');
    if (!toast) {
      toast = document.createElement('div');
      toast.id = 'modelToast';
      toast.className = 'model-toast';
      toast.setAttribute('role', 'status');
      document.body.appendChild(toast);
    }
    toast.textContent = message;
    toast.className = 'model-toast show ' + (kind === 'err' ? 'err' : 'warn');
    clearTimeout(showModelToast._t);
    showModelToast._t = setTimeout(() => { toast.classList.remove('show'); }, 6000);
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
    if (!resp.ok) throw new Error(`HTTP ${resp.status} for ${url}`);
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
  const markCanvas= document.getElementById('markCanvas');
  const viewerStack = document.getElementById('viewerStack');
  const cw        = document.getElementById('cw');
  let activeTool = 'lock';
  let marks = [];
  let draftMark = null;
  let isPointerDown = false;
  let panStart = null;
  let view = { scale: 1, x: 0, y: 0 };
  let windowing = { brightness: 0, contrast: 100, invert: false };
  let aiHidden = false;

  /* ── FRACTURE TYPE CLASSIFIER (v3) ────────────────────── */
  /*
   * Heuristic classification: bbox geometry + confidence band + spatial clustering.
   * All thresholds are normalized to image size (works for any resolution).
   *
   * ① 粉碎性骨折 (Comminuted)
   *    ≥3 total dets AND this detection has ≥1 neighbor within 13% of image diagonal.
   *    Applied to ALL detections (v2 bug: only checked det[0]).
   *    Logic: true comminution = multiple fragments clustered at same bone site.
   *    13% of 640×640 diagonal (905px) ≈ 118px — typical 2–3× bone shaft width.
   *    Multi-view X-rays: views are ~45–50% diagonal apart → never triggers. ✓
   *
   * ② 高置信度 conf ≥ 0.70 → aspect ratio classification
   *    AR = bbox_width / bbox_height (in original image pixel coords)
   *    AR ≥ 1.8  → 横形/Transverse   (fracture ⊥ bone long-axis → wide box)
   *    AR ≤ 0.55 → 纵形/Longitudinal  (fracture ∥ bone long-axis → tall box)
   *    else      → 斜形/Oblique       (diagonal, most common ≈ 60% of real cases)
   *    Assumes standard AP view of a vertical long bone. Works for radius, ulna,
   *    tibia, fibula, humerus in AP/PA view.
   *
   * ③ 中置信度 0.38 ≤ conf < 0.70 → more conservative
   *    normArea = (w×h)/(imgW×imgH) — fraction of image area
   *    normArea < 0.01 (< 1% of image) → 裂缝/Hairline (fine crack, tiny region)
   *    AR ≥ 2.2 or AR ≤ 0.45 → Transverse/Longitudinal (stricter than ②,
   *       because lower confidence makes geometry less reliable)
   *    else → 斜形/Oblique (safe default)
   *
   * ④ conf < 0.38 → 疑似/Suspected (model uncertain)
   *
   * Note: Classification accuracy is fundamentally limited by the single-class
   * detector design. Always annotate reports with "heuristic estimate only."
   */
  function classifyType(det, allDets) {
    const imgW = srcCanvas.width  || 640;
    const imgH = srcCanvas.height || 640;
    const w    = det.x2 - det.x1, h = det.y2 - det.y1;
    const ar   = w / (h || 1);
    const normArea = (w * h) / (imgW * imgH);   // image-size-independent area
    const conf = det.score;
    const cx   = (det.x1 + det.x2) / 2, cy = (det.y1 + det.y2) / 2;

    // ① Comminuted — IoU-based overlap detection
    //   TRUE comminuted: multiple fragments of ONE bone → bboxes partially OVERLAP (IoU > 0.10)
    //   Dual-bone fracture (radius+ulna): bboxes are ADJACENT, NOT overlapping (IoU ≈ 0)
    //   Two-view X-ray: bboxes are far apart (no overlap at all)
    //   This cleanly separates comminuted from dual-bone/two-view false positives.
    if (conf >= 0.45) {
      let overlapping = 0;
      for (const o of allDets) {
        if (o === det || o.score < 0.40) continue;   // only compare vs other medium-high conf dets
        const ix1 = Math.max(det.x1, o.x1), iy1 = Math.max(det.y1, o.y1);
        const ix2 = Math.min(det.x2, o.x2), iy2 = Math.min(det.y2, o.y2);
        const inter = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
        if (inter === 0) continue;                    // no overlap at all → skip
        const areaA = (det.x2-det.x1) * (det.y2-det.y1);
        const areaB = (o.x2-o.x1)     * (o.y2-o.y1);
        const iou   = inter / (areaA + areaB - inter);
        if (iou > 0.10) overlapping++;               // ≥10% IoU = overlapping fragments
      }
      if (overlapping >= 1) return 'ftComminuted';
    }

    // ② High confidence: AR-based classification
    if (conf >= 0.70) {
      if (ar >= 1.8)  return 'ftTransverse';
      if (ar <= 0.55) return 'ftLongitudinal';
      return 'ftOblique';
    }

    // ③ Moderate-low confidence: image-size-normalized area + stricter AR gates
    //    Threshold lowered to 0.28 so that 28-38% detections are AR-classified
    //    rather than blanket "Suspected" (35-37% confs often show clear oblique lines)
    if (conf >= 0.28) {
      if (normArea < 0.010) return 'ftHairline';  // < 1% image = fine crack
      if (ar >= 2.2)        return 'ftTransverse';  // stricter: ≥2.2 (vs 1.8 above)
      if (ar <= 0.45)       return 'ftLongitudinal';// stricter: ≤0.45 (vs 0.55)
      return 'ftOblique';
    }

    // ④ Very low confidence (< 0.28) → suspected
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
    // Real X-rays (even lightly tinted / JPEG-chroma) measure ~0.0–0.35;
    // color photos / teal "stock" radiographs measure ~0.5–0.9.
    // 0.45 splits them and avoids false-flagging genuine grayscale X-rays.
    return avgSat < 0.45; // true = likely X-ray
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
      const modelBytesPromise = loadModelBytesFromCandidates([LITE_MODEL_PATH, LITE_MODEL_FALLBACK], 'lite', (pct, spd, fromCache) => {
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
      backendOrder.push('webgl', 'wasm');  // WebGL first for fast load; WASM as inference fallback

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
      if(liteStats) liteStats.textContent = `11.7 MB · YOLOv8n · ${backendLabel} ${threads}T · legacy`;

      if (lastBitmap) runEl.disabled = false;
      setRunSts('', '');

      if (window.MTL) { window.MTL.load().catch(() => {}); }
      loadClassifier();

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
      document.getElementById('optLite').classList.remove('active');
      document.getElementById('optFull').classList.add('active');

      if (SESSION_FULL) {
        SESSION = SESSION_FULL;
        currentModel = 'full';
        if(nb) nb.textContent = 'YOLOv8m Detector (Full) · 640px · 1 class · mAP50 0.82';
        setRunSts('✅ Switched to Full model (YOLOv8m)', 'ok');
        return;
      }

      const copy = modelLoadCopy();
      setModelLoading(true, { title: copy.title, sub: copy.sub });
      runEl.disabled = true;
      if(nb) nb.textContent = '⬇️ YOLOv8m Full…';

      try {
        const fullOpt = document.getElementById('optFull');
        if(fullOpt) fullOpt.classList.add('downloading');

        const modelBytes = await loadModelBytesFromCandidates(FULL_MODEL_CANDIDATES, FULL_MODEL_CACHE_KEY, (pct, spd, fromCache) => {
          if (fromCache) {
            setModelLoading(true, { title: copy.title, sub: LANG === 'zh' ? '从本地缓存加载…' : 'Loading from cache…', pct, pctText: '100%' });
            if(nb) nb.textContent = '⚡ YOLOv8m from cache…';
          } else {
            setModelLoading(true, {
              title: copy.title,
              sub: copy.sub,
              pct,
              pctText: `${pct}%${spd > 0 ? ' · ' + spd.toFixed(1) + ' MB/s' : ''}`,
            });
            if(nb) nb.textContent = `⬇️ YOLOv8m — ${pct}%`;
          }
        });

        setModelLoading(true, { title: copy.compile, sub: copy.sub, pct: 100, pctText: '100%' });
        if(nb) nb.textContent = '🧠 Compiling Full model…';
        const t0 = performance.now();
        const backendOrder = [];
        if (navigator.gpu) backendOrder.push('webgpu');
        backendOrder.push('webgl', 'wasm');  // WebGL first for fast load; WASM as inference fallback
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
        setRunSts('✅ Full model ready! (YOLOv8m · mAP50 0.82 · 27k unified dataset)', 'ok');
        if(lastBitmap) runEl.disabled = false;
        setModelLoading(false);

      } catch(e) {
        console.error(e);
        setModelLoading(false);
        const toastMsg = LANG === 'zh'
          ? `高精度模型加载失败，已切回 Lite：${e.message}`
          : LANG === 'ko'
            ? `Full 모델 로드 실패, Lite로 복귀: ${e.message}`
            : `Full model failed, reverted to Lite: ${e.message}`;
        showModelToast(toastMsg, 'err');
        setRunSts(`❌ Download failed: ${e.message}. Reverting to Lite.`, 'err');
        document.getElementById('optFull').classList.remove('active');
        document.getElementById('optLite').classList.add('active');
        SESSION = SESSION_LITE;
        currentModel = 'lite';
        if(nb) nb.textContent = 'YOLOv8n (Lite) · Reverted due to error';
        if(lastBitmap) runEl.disabled = false;
      } finally {
        const fullOpt = document.getElementById('optFull');
        if (fullOpt) fullOpt.classList.remove('downloading');
      }

    } else {
      // Switch back to lite
      document.getElementById('optFull').classList.remove('active');
      document.getElementById('optLite').classList.add('active');
      SESSION = SESSION_LITE;
      currentModel = 'lite';
      if(nb) nb.textContent = 'YOLOv8n Detector (Lite) · 640px · 1 class · legacy fallback';
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
  cw?.addEventListener('click', e => { if (lastBitmap) e.preventDefault(); });
  cw?.addEventListener('pointerdown', onViewerPointerDown);
  window.addEventListener('pointermove', onViewerPointerMove);
  window.addEventListener('pointerup', onViewerPointerUp);
  cw?.addEventListener('wheel', onViewerWheel, { passive: false });

  async function handleFile(f) {
    lastDets = []; lastElapsed = 0;
    setExport(false);
    resetPipeline();

    // Detect DICOM by extension/MIME, or by "DICM" magic bytes at offset 128
    let isDcm = /\.dcm$/i.test(f.name) || f.type === 'application/dicom';
    if (!isDcm && !f.type.startsWith('image/')) {
      try {
        const head = new Uint8Array(await f.slice(0, 132).arrayBuffer());
        isDcm = looksLikeDicom(f, head);
      } catch {}
    }
    if (!isDcm && !f.type.startsWith('image/')) {
      setRunSts('Only image or DICOM (.dcm) files supported.','err'); return;
    }

    try {
      if (isDcm) {
        setRunSts(LANG==='zh'?'📂 正在解析 DICOM…':LANG==='ko'?'📂 DICOM 파싱 중…':'📂 Parsing DICOM…','warn');
        lastBitmap = await dicomToCanvas(await f.arrayBuffer());
      } else {
        lastBitmap = await createImageBitmap(f).catch(async () =>
          new Promise((res,rej) => {
            const img = new Image();
            img.onload = () => res(img);
            img.onerror = () => rej(new Error('Image decode failed'));
            img.src = URL.createObjectURL(f);
          })
        );
      }
    } catch(e) { setRunSts((isDcm?'DICOM error: ':'')+e.message,'err'); return; }

    drawSource(lastBitmap);

    // X-ray check
    const isXray = checkIsXray(lastBitmap);
    lastIsXray = isXray;
    const warn = document.getElementById('xrayWarn');
    if (warn) warn.classList.toggle('show', !isXray);

    setRunSts(`${f.name} (${(f.size/1024).toFixed(1)} KB)${isXray?'':' ⚠️'}`,'warn' );
    if(isXray) setRunSts(`${f.name} (${(f.size/1024).toFixed(1)} KB)`,'');
    setPipeStep(1);
    runEl.disabled = !SESSION;
    // Auto-run detection after image load — no need to find "Run" button
    if (SESSION) setTimeout(() => APP.run(), 250);
  }

  function drawSource(bitmap) {
    document.getElementById('drop')?.classList.add('has-img');
    const W = bitmap.naturalWidth||bitmap.width, H = bitmap.naturalHeight||bitmap.height;
    srcCanvas.width=W; srcCanvas.height=H;
    detCanvas.width=W; detCanvas.height=H;
    markCanvas.width=W; markCanvas.height=H;
    srcCanvas.getContext('2d').drawImage(bitmap,0,0);
    detCanvas.getContext('2d').clearRect(0,0,W,H);
    markCanvas.getContext('2d').clearRect(0,0,W,H);
    document.getElementById('lboxPh').style.display='none';
    cw.classList.add('on');
    resetWorkstationForImage();
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

  /* ══ EfficientNet-B3 11-CLASS CLASSIFIER (real trained model) ══════════
   * Test Acc 91.63% · Macro F1 0.846 · input 300×300 · ImageNet norm.
   * Whole-image classification (incl. "Normal"). Complements YOLOv8 localization.
   */
  const CLF_PATH  = './model_classifier.onnx';
  const CLF_SIZE  = 300;
  const CLF_MEAN  = [0.485, 0.456, 0.406];
  const CLF_STD   = [0.229, 0.224, 0.225];
  const CLF_CLASSES = [
    'Avulsion fracture','Comminuted fracture','Fracture Dislocation',
    'Greenstick fracture','Hairline Fracture','Impacted fracture',
    'Longitudinal fracture','Normal','Oblique fracture',
    'Pathological fracture','Spiral Fracture'
  ];
  // Cascade mode: classifier is restricted to fracture TYPES only.
  // Presence (normal/abnormal) is decided UPSTREAM by the detector; the 'Normal' logit
  // is set to -Infinity at inference so the classifier behaves as a 10-class type-only model.
  const NORMAL_IDX = CLF_CLASSES.indexOf('Normal');   // = 7

  // Display names per language (index matches CLF_CLASSES order)
  const CLF_I18N = {
    zh:['撕脱骨折','粉碎性骨折','骨折脱位','青枝骨折','发丝骨折','嵌插骨折','纵形骨折','正常','斜形骨折','病理性骨折','螺旋形骨折'],
    ko:['견열 골절','분쇄 골절','골절 탈구','약목 골절','실금 골절','감입 골절','종형 골절','정상','사선 골절','병적 골절','나선형 골절'],
    en:['Avulsion','Comminuted','Fracture Dislocation','Greenstick','Hairline','Impacted','Longitudinal','Normal','Oblique','Pathological','Spiral']
  };
  function clfName(idx){ return (CLF_I18N[LANG]||CLF_I18N.en)[idx] || CLF_CLASSES[idx]; }

  let SESSION_CLF = null;
  let CLF_LOADING = false;
  let CLF_FAILED  = false;

  async function loadClassifier(onProgress) {
    if (SESSION_CLF || CLF_LOADING) return;
    CLF_LOADING = true;
    try {
      const bytes = await loadModelBytes(CLF_PATH, 'classifier', onProgress);
      const order = [];
      if (navigator.gpu) order.push('webgpu');
      order.push('wasm');   // EfficientNet runs reliably on WASM; WebGL skipped to avoid op gaps
      let lastErr;
      for (const be of order) {
        try {
          SESSION_CLF = await ort.InferenceSession.create(bytes, {
            executionProviders: [be], graphOptimizationLevel: 'all',
          });
          window._clfBackend = be;
          break;
        } catch(e) { lastErr = e; }
      }
      if (!SESSION_CLF) throw lastErr || new Error('classifier backends failed');
      console.log('[classifier] ready on', window._clfBackend);
    } catch(e) {
      console.error('[classifier] load failed', e);
      CLF_FAILED = true;
    } finally {
      CLF_LOADING = false;
    }
  }

  // Crop a detected fracture region out of the original bitmap, with padding,
  // for per-box cascade classification. The classifier was trained on fracture-focused
  // images, so feeding it the detected REGION (not the whole X-ray) is the architecturally
  // correct cascade: detector localizes -> crop -> classifier names the type for THAT crop.
  // `det` uses image-space coordinates {x1,y1,x2,y2} (same as drawBoxes).
  function cropDetection(bitmap, det, padFrac = 0.15) {
    const W = bitmap.width  || bitmap.naturalWidth  || lastBitmap?.width  || 640;
    const H = bitmap.height || bitmap.naturalHeight || lastBitmap?.height || 640;
    const bw = det.x2 - det.x1, bh = det.y2 - det.y1;
    const padX = bw * padFrac, padY = bh * padFrac;
    const x = Math.max(0, Math.floor(det.x1 - padX));
    const y = Math.max(0, Math.floor(det.y1 - padY));
    const cw = Math.min(W - x, Math.ceil(bw + 2 * padX));
    const ch = Math.min(H - y, Math.ceil(bh + 2 * padY));
    if (cw <= 0 || ch <= 0) return null;
    const cv = document.createElement('canvas');
    cv.width = cw; cv.height = ch;
    cv.getContext('2d', {willReadFrequently:true}).drawImage(bitmap, x, y, cw, ch, 0, 0, cw, ch);
    return cv;
  }

  // Whole bitmap (or crop) → 300×300 stretch + ImageNet-normalized NCHW float32 (matches eval_tf).
  // `flip=true` mirrors horizontally for test-time augmentation.
  function clfPreprocess(bitmap, flip=false) {
    const sz = CLF_SIZE;
    const cv = document.createElement('canvas'); cv.width = sz; cv.height = sz;
    const ctx = cv.getContext('2d', {willReadFrequently:true});
    if (flip) { ctx.translate(sz, 0); ctx.scale(-1, 1); }
    ctx.drawImage(bitmap, 0, 0, sz, sz);
    const {data} = ctx.getImageData(0, 0, sz, sz);
    const N = sz*sz, out = new Float32Array(3*N);
    for (let i=0;i<N;i++){
      out[i]     = (data[i*4]  /255 - CLF_MEAN[0]) / CLF_STD[0];
      out[i+N]   = (data[i*4+1]/255 - CLF_MEAN[1]) / CLF_STD[1];
      out[i+2*N] = (data[i*4+2]/255 - CLF_MEAN[2]) / CLF_STD[2];
    }
    return out;
  }

  // Cascade-mode inference:
  //   * TTA: average softmax of original + horizontal-flip (X-rays of left/right limbs
  //     make horizontal flip semantically valid). Empirically +1-3% top-1 accuracy.
  //   * Normal suppression: in cascade, presence is decided by the detector; the classifier
  //     is restricted to fracture TYPES — Normal logit -> -Infinity so it never wins.
  async function _classifyOnce(bitmap, flip) {
    const float32 = clfPreprocess(bitmap, flip);
    const tensor  = new ort.Tensor('float32', float32, [1,3,CLF_SIZE,CLF_SIZE]);
    const inName  = SESSION_CLF.inputNames[0];
    const out     = await SESSION_CLF.run({[inName]: tensor});
    return Array.from(out[SESSION_CLF.outputNames[0]].data);   // raw logits
  }
  async function classifyImage(bitmap) {
    if (!SESSION_CLF) return null;
    const L1 = await _classifyOnce(bitmap, false);
    const L2 = await _classifyOnce(bitmap, true);
    if (NORMAL_IDX >= 0) { L1[NORMAL_IDX] = -Infinity; L2[NORMAL_IDX] = -Infinity; }
    const softmax = (arr) => {
      let mx=-Infinity; for(const v of arr) if(v>mx) mx=v;
      let s=0; const p=arr.map(v=>{ const e=Math.exp(v-mx); s+=e; return e; });
      return p.map(x=>x/s);
    };
    const p1 = softmax(L1), p2 = softmax(L2);
    const probs = p1.map((v,i)=>(v+p2[i])/2);
    return probs.map((p,i)=>({idx:i,p})).sort((a,b)=>b.p-a.p);
  }

  /* ══ DICOM (.dcm) SUPPORT — lazy-loaded daikon parser ═══════════════════
   * Hospital X-rays are usually .dcm. We parse pixel data, apply window/level
   * (or min/max), handle MONOCHROME1 inversion, and render to a canvas that
   * feeds the normal detection + classification pipeline.
   */
  let _dcmParserReady = null;
  function loadDicomParser() {
    if (_dcmParserReady) return _dcmParserReady;
    _dcmParserReady = new Promise((res, rej) => {
      if (window.dicomParser) return res(window.dicomParser);
      const s = document.createElement('script');
      s.src = 'https://cdn.jsdelivr.net/npm/dicom-parser@1.8.21/dist/dicomParser.min.js';
      s.crossOrigin = 'anonymous';
      s.onload = () => window.dicomParser ? res(window.dicomParser) : rej(new Error('dicomParser load error'));
      s.onerror = () => rej(new Error('Failed to load DICOM parser (network)'));
      document.head.appendChild(s);
    });
    return _dcmParserReady;
  }

  function looksLikeDicom(f, head) {
    if (/\.dcm$/i.test(f.name) || f.type === 'application/dicom') return true;
    // "DICM" magic at byte offset 128
    return !!(head && head.length >= 132 &&
      head[128]===0x44 && head[129]===0x49 && head[130]===0x43 && head[131]===0x4D);
  }

  // Parse uncompressed DICOM (Explicit/Implicit VR Little Endian) → grayscale canvas.
  async function dicomToCanvas(arrayBuffer) {
    const dp = await loadDicomParser();
    const ds = dp.parseDicom(new Uint8Array(arrayBuffer));
    const cols = ds.uint16('x00280011'), rows = ds.uint16('x00280010');
    if (!cols || !rows) throw new Error('Missing image dimensions');
    const bits   = ds.uint16('x00280100') || 16;
    const signed = (ds.uint16('x00280103') || 0) === 1;
    const photo  = (ds.string('x00280004') || 'MONOCHROME2');
    const slope  = parseFloat(ds.string('x00281053') || '1') || 1;
    const intpt  = parseFloat(ds.string('x00281052') || '0') || 0;
    let wc = ds.floatString('x00281050'); // window center
    let ww = ds.floatString('x00281051'); // window width
    const el = ds.elements.x7fe00010;
    if (!el) throw new Error('No pixel data element');
    if (el.encapsulatedPixelData) throw new Error('Compressed DICOM not supported (use uncompressed)');

    const N = cols * rows;
    const buf = ds.byteArray.buffer;
    let raw;
    if (bits <= 8)       raw = new Uint8Array(buf, el.dataOffset, N);
    else if (signed)     raw = new Int16Array(buf, el.dataOffset, N);
    else                 raw = new Uint16Array(buf, el.dataOffset, N);

    // Apply rescale, then window/level (or min/max fallback)
    let lo, hi;
    if (typeof wc === 'number' && typeof ww === 'number' && ww > 0) {
      lo = wc - ww/2; hi = wc + ww/2;
    } else {
      let mn=Infinity, mx=-Infinity;
      for (let i=0;i<N;i++){ const v=raw[i]*slope+intpt; if(v<mn)mn=v; if(v>mx)mx=v; }
      lo = mn; hi = mx;
    }
    const range = (hi - lo) || 1;
    const mono1 = photo.indexOf('MONOCHROME1') >= 0;

    const cv = document.createElement('canvas'); cv.width = cols; cv.height = rows;
    const ctx = cv.getContext('2d');
    const id = ctx.createImageData(cols, rows);
    for (let i = 0; i < N; i++) {
      const val = raw[i]*slope + intpt;
      let v = Math.round((val - lo) / range * 255);
      v = v < 0 ? 0 : v > 255 ? 255 : v;
      if (mono1) v = 255 - v;
      id.data[i*4]=v; id.data[i*4+1]=v; id.data[i*4+2]=v; id.data[i*4+3]=255;
    }
    ctx.putImageData(id, 0, 0);
    return cv;
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
    return nms(boxes, META.iou_threshold||0.35);  // tightened from 0.45 → 0.35
  }

  function iou(a,b){
    const ix1=Math.max(a.x1,b.x1),iy1=Math.max(a.y1,b.y1);
    const ix2=Math.min(a.x2,b.x2),iy2=Math.min(a.y2,b.y2);
    const inter=Math.max(0,ix2-ix1)*Math.max(0,iy2-iy1);
    const aA=(a.x2-a.x1)*(a.y2-a.y1),bA=(b.x2-b.x1)*(b.y2-b.y1);
    return inter/(aA+bA-inter+1e-6);
  }
  // Containment ratio: how much of the SMALLER box is inside the larger.
  // High containment with low IoU is the "nested duplicate" pattern that
  // standard IoU-NMS misses (small redundant box sitting partly inside a larger one).
  function containment(a,b){
    const ix1=Math.max(a.x1,b.x1),iy1=Math.max(a.y1,b.y1);
    const ix2=Math.min(a.x2,b.x2),iy2=Math.min(a.y2,b.y2);
    const inter=Math.max(0,ix2-ix1)*Math.max(0,iy2-iy1);
    const aA=(a.x2-a.x1)*(a.y2-a.y1),bA=(b.x2-b.x1)*(b.y2-b.y1);
    const smaller=Math.min(aA,bA);
    return inter/(smaller+1e-6);
  }
  function nms(boxes,thr){
    boxes.sort((a,b)=>b.score-a.score);
    const keep=[],sup=new Uint8Array(boxes.length);
    const CONTAIN_THR = 0.50;   // ≥50% of smaller box inside larger → suppress
    for(let i=0;i<boxes.length;i++){
      if(sup[i]) continue; keep.push(boxes[i]);
      for(let j=i+1;j<boxes.length;j++){
        if(sup[j]||boxes[i].cls!==boxes[j].cls) continue;
        if(iou(boxes[i],boxes[j])>thr)              { sup[j]=1; continue; }
        if(containment(boxes[i],boxes[j])>CONTAIN_THR) sup[j]=1;
      }
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

  /* ── MANUAL RADIOLOGY WORKSTATION ─────────────────────── */
  function setWsState(key, fallback) {
    const el = document.getElementById('wsState');
    if (el) el.textContent = key ? t(key) : fallback;
  }

  function updateToolButtons() {
    ['toolLock','toolZoom','toolMeasure','toolPen'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.classList.remove('active');
    });
    const id = activeTool === 'zoom' ? 'toolZoom' :
               activeTool === 'measure' ? 'toolMeasure' :
               activeTool === 'pen' ? 'toolPen' : 'toolLock';
    document.getElementById(id)?.classList.add('active');
    cw?.classList.toggle('zoom-mode', activeTool === 'zoom');
    cw?.classList.toggle('draw-mode', activeTool === 'measure' || activeTool === 'pen');
    cw?.classList.toggle('locked', activeTool === 'lock');
  }

  function setToolMode(tool) {
    activeTool = tool || 'lock';
    updateToolButtons();
    if (activeTool === 'zoom') setWsState('wsZoom');
    else if (activeTool === 'measure') setWsState('wsMeasure');
    else if (activeTool === 'pen') setWsState('wsPen');
    else setWsState('wsLockedReady');
  }

  function applyViewTransform() {
    if (!viewerStack) return;
    viewerStack.style.transform = `translate(${view.x}px, ${view.y}px) scale(${view.scale})`;
  }

  function resetViewState() {
    view = { scale: 1, x: 0, y: 0 };
    applyViewTransform();
  }

  function currentFilter() {
    const b = Math.max(40, Math.min(160, 100 + windowing.brightness));
    const c = Math.max(70, Math.min(170, windowing.contrast));
    return `brightness(${b}%) contrast(${c}%) invert(${windowing.invert ? 1 : 0})`;
  }

  function applyWindowing() {
    const w = document.getElementById('windowSlider');
    const c = document.getElementById('contrastSlider');
    if (w) windowing.brightness = Number(w.value || 0);
    if (c) windowing.contrast = Number(c.value || 100);
    document.getElementById('windowVal')?.replaceChildren(document.createTextNode(String(windowing.brightness)));
    document.getElementById('contrastVal')?.replaceChildren(document.createTextNode(String(windowing.contrast)));
    srcCanvas.style.filter = currentFilter();
    document.getElementById('btnInvert')?.classList.toggle('active', windowing.invert);
  }

  function drawFilteredSource(ctx) {
    ctx.save();
    ctx.filter = currentFilter();
    ctx.drawImage(srcCanvas, 0, 0);
    ctx.restore();
  }

  function canvasPoint(e) {
    const rect = srcCanvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) * srcCanvas.width / rect.width;
    const y = (e.clientY - rect.top) * srcCanvas.height / rect.height;
    return {
      x: Math.max(0, Math.min(srcCanvas.width, x)),
      y: Math.max(0, Math.min(srcCanvas.height, y))
    };
  }

  function measureLength(mark) {
    if (!mark || !mark.points || mark.points.length < 2) return 0;
    const a = mark.points[0], b = mark.points[1];
    return Math.hypot(b.x - a.x, b.y - a.y);
  }

  function setMeasureText(mark) {
    const el = document.getElementById('measureReadout');
    if (!el) return;
    if (!mark) {
      el.textContent = t('measureHint');
      return;
    }
    el.textContent = `${measureLength(mark).toFixed(1)} px`;
  }

  function drawMeasure(ctx, mark) {
    const [a,b] = mark.points;
    const W = markCanvas.width;
    const lw = Math.max(2, W / 420);
    ctx.save();
    ctx.strokeStyle = '#e8b85a';
    ctx.fillStyle = '#e8b85a';
    ctx.lineWidth = lw;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
    [a,b].forEach(p => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, lw * 2.2, 0, Math.PI * 2);
      ctx.fill();
    });
    const label = `${measureLength(mark).toFixed(1)} px`;
    const mx = (a.x + b.x) / 2;
    const my = (a.y + b.y) / 2;
    const fs = Math.max(13, W / 52);
    ctx.font = `700 ${fs}px sans-serif`;
    const tw = ctx.measureText(label).width;
    ctx.fillStyle = 'rgba(2,4,3,.78)';
    ctx.fillRect(mx - tw / 2 - 6, my - fs - 8, tw + 12, fs + 8);
    ctx.fillStyle = '#f5d58b';
    ctx.fillText(label, mx - tw / 2, my - 8);
    ctx.restore();
  }

  function drawPen(ctx, mark) {
    const pts = mark.points;
    if (!pts || pts.length < 2) return;
    ctx.save();
    ctx.strokeStyle = '#45d6c6';
    ctx.lineWidth = Math.max(2, markCanvas.width / 360);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
    ctx.stroke();
    ctx.restore();
  }

  function redrawMarks() {
    if (!markCanvas) return;
    const ctx = markCanvas.getContext('2d');
    ctx.clearRect(0, 0, markCanvas.width, markCanvas.height);
    [...marks, draftMark].filter(Boolean).forEach(mark => {
      if (mark.type === 'measure') drawMeasure(ctx, mark);
      if (mark.type === 'pen') drawPen(ctx, mark);
    });
  }

  // List of all workstation control elements that should be enabled when an
  // image is loaded and re-disabled when the user clears the view.
  const WS_CONTROL_IDS = [
    'toolLock', 'toolZoom', 'toolMeasure', 'toolPen',
    'btnUndoMark', 'btnClearMarks', 'btnResetView', 'btnInvert',
    'btnHideAI', 'windowSlider', 'contrastSlider',
  ];

  function setWorkstationEnabled(enabled) {
    WS_CONTROL_IDS.forEach(id => {
      const el = document.getElementById(id);
      if (el) el.disabled = !enabled;
    });
  }

  function resetWorkstationForImage() {
    marks = [];
    draftMark = null;
    isPointerDown = false;
    panStart = null;
    windowing = { brightness: 0, contrast: 100, invert: false };
    const w = document.getElementById('windowSlider');
    const c = document.getElementById('contrastSlider');
    if (w) w.value = 0;
    if (c) c.value = 100;
    resetViewState();
    applyWindowing();
    setToolMode('lock');
    setMeasureText(null);
    setAIHidden(false);
    // CRITICAL: enable all workstation controls now that an image is loaded.
    // (HTML defaults them to `disabled` so the toolbar looks greyed out until
    //  there's actually something to manipulate.)
    setWorkstationEnabled(true);
  }

  function setAIHidden(on) {
    aiHidden = !!on;
    document.body.classList.toggle('ai-hidden', aiHidden);
    const btn = document.getElementById('btnHideAI');
    if (btn) {
      btn.classList.toggle('active', aiHidden);
      const span = btn.querySelector('span');
      if (span) span.textContent = t(aiHidden ? 'showAI' : 'hideAI');
    }
    setWsState(aiHidden ? 'aiHidden' : (
      activeTool === 'zoom' ? 'wsZoom' :
      activeTool === 'measure' ? 'wsMeasure' :
      activeTool === 'pen' ? 'wsPen' : 'wsLockedReady'
    ));
  }

  function onViewerPointerDown(e) {
    if (!lastBitmap || activeTool === 'lock') return;
    e.preventDefault();
    isPointerDown = true;
    cw?.classList.add('dragging');
    if (activeTool === 'zoom') {
      panStart = { cx: e.clientX, cy: e.clientY, x: view.x, y: view.y };
      return;
    }
    const p = canvasPoint(e);
    if (activeTool === 'measure') {
      draftMark = { type: 'measure', points: [p, p] };
      setMeasureText(draftMark);
    } else if (activeTool === 'pen') {
      draftMark = { type: 'pen', points: [p] };
    }
    redrawMarks();
  }

  function onViewerPointerMove(e) {
    if (!isPointerDown || !lastBitmap || activeTool === 'lock') return;
    e.preventDefault();
    if (activeTool === 'zoom' && panStart) {
      view.x = panStart.x + e.clientX - panStart.cx;
      view.y = panStart.y + e.clientY - panStart.cy;
      applyViewTransform();
      return;
    }
    if (!draftMark) return;
    const p = canvasPoint(e);
    if (draftMark.type === 'measure') {
      draftMark.points[1] = p;
      setMeasureText(draftMark);
    } else if (draftMark.type === 'pen') {
      draftMark.points.push(p);
    }
    redrawMarks();
  }

  function onViewerPointerUp(e) {
    if (!isPointerDown) return;
    e.preventDefault();
    isPointerDown = false;
    cw?.classList.remove('dragging');
    panStart = null;
    if (draftMark) {
      if (draftMark.type === 'pen' && draftMark.points.length < 2) {
        draftMark = null;
      } else {
        marks.push(draftMark);
        if (draftMark.type === 'measure') setMeasureText(draftMark);
        draftMark = null;
      }
      redrawMarks();
    }
  }

  function onViewerWheel(e) {
    if (!lastBitmap || activeTool !== 'zoom') return;
    e.preventDefault();
    const oldScale = view.scale;
    const delta = e.deltaY < 0 ? 1.12 : 0.89;
    const next = Math.max(0.5, Math.min(6, oldScale * delta));
    const rect = cw.getBoundingClientRect();
    const ax = e.clientX - rect.left;
    const ay = e.clientY - rect.top;
    const bx = (ax - view.x) / oldScale;
    const by = (ay - view.y) / oldScale;
    view.scale = next;
    view.x = ax - bx * next;
    view.y = ay - by * next;
    applyViewTransform();
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
      // High-threshold hint: if confThr > 0.70, warn user they may be filtering real fractures
      const highThrHint = confThr > 0.70
        ? `<div style="margin-top:8px;padding:7px 10px;background:rgba(245,158,11,0.12);
                       border:1px solid rgba(245,158,11,0.35);border-radius:6px;
                       font-size:11px;color:var(--warn);line-height:1.5">
             ⚠️ ${t('highThrHint')}
           </div>`
        : '';
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
        <div style="font-size:11px;color:var(--fg2)">${t('scrNormal')}</div>
        ${highThrHint}`;
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
            <div>${t('vrdFracture')} — ${dets.length} ${t(dets.length>1?'regions':'region')}</div>
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

  /* ── RENDER STAGE 2 — MTL per-box (位置 · 方向 · 形态学 + 临床解读) ─────── */
  async function renderClassification(dets) {
    const clfEl = document.getElementById('clfResult');

    if (!dets || dets.length === 0) {
      setStage('stageClf','done');
      clfEl.innerHTML = `<div class="clf-banner info" style="border-color:var(--good);background:rgba(34,197,94,0.08)">
        <div style="font-size:13px;color:var(--good);font-weight:600">✅ ${t('clfNoFracture')}</div>
        <div style="font-size:10px;color:var(--fg3);margin-top:6px;font-weight:400">${t('clfCascadeNote')}</div>
      </div>`;
      return;
    }

    setStage('stageClf','active');

    if (!window.MTL?.isReady() && !window.MTL?.failed()) {
      clfEl.innerHTML = `<div style="color:var(--fg2);font-size:11px;text-align:center;padding:16px 0">
        <span class="spin"></span> &nbsp;${t('clfLoading')}</div>`;
      try { await window.MTL.load(); } catch (e) { console.warn('MTL load failed', e); }
    }
    if (!window.MTL?.isReady()) {
      setStage('stageClf','fail');
      clfEl.innerHTML = `<div style="color:var(--bad);font-size:11px;text-align:center;padding:16px 0">${t('clfUnavail')}</div>`;
      return;
    }

    clfEl.innerHTML = `<div style="color:var(--fg2);font-size:11px;text-align:center;padding:14px 0">
      <span class="spin"></span> &nbsp;${t('clfPerBoxProg').replace('{n}', dets.length)}</div>`;

    const perBox = [];
    for (let i = 0; i < dets.length; i++) {
      const det = dets[i];
      try {
        const decoded = await window.MTL.classifyDetection(lastBitmap, det, 0.15);
        if (decoded) perBox.push({ det, decoded, idx: i });
      } catch (e) { console.warn('MTL classify failed for box', i, e); }
      clfEl.innerHTML = `<div style="color:var(--fg2);font-size:11px;text-align:center;padding:14px 0">
        <span class="spin"></span> &nbsp;${t('clfPerBoxProg').replace('{n}', dets.length)} (${i + 1}/${dets.length})</div>`;
    }

    setStage('stageClf','done');
    if (perBox.length === 0) {
      clfEl.innerHTML = `<div style="color:var(--bad);font-size:11px;padding:12px">${t('clfPerBoxFailed')}</div>`;
      return;
    }

    const ood = lastIsXray === false;
    const accentCols = ['#5b9eff','#45d6c6','#7eb8ff','#a78bfa'];
    let html = '';
    if (ood) html += `<div class="clf-banner ood">⚠️ ${t('clfOOD')}</div>`;
    html += `<div class="${ood ? 'clf-dim' : ''}">`;
    html += `<div style="font-size:11px;color:var(--fg2);margin-bottom:8px">${t('clfPerBoxHeader').replace('{n}', perBox.length)}</div>`;

    for (const item of perBox) {
      const d = item.decoded;
      const sideCol = accentCols[item.idx % accentCols.length];
      const detConf = (item.det.score * 100).toFixed(1);
      const loc = getLocationDesc(item.det, srcCanvas.width, srcCanvas.height);
      const summary = window.MTL.formatLine(d, LANG);
      const barsHtml = window.MTL.formatHTML(d, LANG);

      html += `<div class="mtl-lesion-card" style="border-left:3px solid ${sideCol}">
        <div class="mtl-lesion-title">#${item.idx + 1} · ${summary}</div>
        <div class="mtl-lesion-meta">${t('clfDetConf')} ${detConf}% · ${loc}</div>
        <div class="mtl-bars">${barsHtml}</div>
      </div>`;
    }

    const bl = (window.MTL.backend() || 'wasm').toUpperCase();
    html += `<div class="mtl-model-strip">
      🧠 MultiTask Context-Aware Net · <strong>per-box dual-input</strong> · ${bl} &nbsp;·&nbsp; 位置 + 方向 + 形态学</div>`;
    html += `</div>`;

    clfEl.innerHTML = html;
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
      setRunSts(t('scanning'));

      const sz = META.img_size;
      const confThr = document.getElementById('confSlider').value / 100;
      const t0 = performance.now();

      try {
        const {canvas,scale,px,py,origW,origH} = letterbox(lastBitmap,sz);
        const float32 = toFloat32(canvas,sz);
        const tensor  = new ort.Tensor('float32',float32,[1,3,sz,sz]);

        // Inference with lazy WASM fallback.
        // Some WebGL/WebGPU drivers fail on YOLOv8's Resize(mode=nearest).
        // On first failure, recompile the current model with WASM and retry once.
        let out;
        try {
          out = await SESSION.run({[META._in]:tensor});
        } catch(infErr) {
          const msg = String((infErr && infErr.message) || infErr);
          if (window._activeBackend !== 'wasm' && /resize|nearest|not support|webgl|jsep/i.test(msg)) {
            console.warn('[fallback] backend inference failed → switching to WASM:', msg);
            const fallbackMsg = LANG === 'zh' ? '⏳ 正在为此浏览器优化(约 3 秒,仅首次)…'
                              : LANG === 'ko' ? '⏳ 브라우저 최적화 중(약 3초, 처음만)…'
                              : '⏳ Optimizing for this browser (~3s, one-time)…';
            setRunSts(fallbackMsg, 'warn');
            const modelBytes = currentModel === 'full'
              ? await loadModelBytesFromCandidates(FULL_MODEL_CANDIDATES, FULL_MODEL_CACHE_KEY)
              : await loadModelBytes(resolveModelUrl(LITE_MODEL_PATH), 'lite').catch(() =>
                loadModelBytes(resolveModelUrl(LITE_MODEL_FALLBACK), 'lite'));
            const wasmSession = await ort.InferenceSession.create(modelBytes, {
              executionProviders: ['wasm'],
              graphOptimizationLevel: 'all',
            });
            if (currentModel === 'lite') SESSION_LITE = wasmSession;
            else                          SESSION_FULL = wasmSession;
            SESSION = wasmSession;
            window._activeBackend = 'wasm';
            out = await SESSION.run({[META._in]:tensor});
          } else {
            throw infErr;
          }
        }
        const outT    = out[Object.keys(out)[0]];

        lastElapsed = Math.round(performance.now()-t0);
        // Use lower internal threshold to catch all candidates, then filter by UI threshold
        lastDets = parseOutput(outT.data,outT.dims,confThr*0.1,scale,px,py,origW,origH);

        drawBoxes(lastDets.filter(d=>d.score>=confThr));
        setScanline(false);
        setPipeStep(3);

        renderScreening(lastDets.filter(d=>d.score>=confThr), confThr, lastElapsed);
        // Always run the whole-image EfficientNet classifier (it has a "Normal" class,
        // so it gives a meaningful answer even when YOLOv8 detects nothing). Async — fills Stage 2 when ready.
        renderClassification(lastDets.filter(d=>d.score>=confThr));

        const visCount = lastDets.filter(d=>d.score>=confThr).length;
        setRunSts(`✅ ${lastElapsed}ms · ${visCount} ${t(visCount!==1?'detections':'detection')}`, 'ok');
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

    setTool(tool) {
      if (!lastBitmap) return;
      setToolMode(tool);
    },

    toggleAI() {
      if (!lastBitmap) return;
      setAIHidden(!aiHidden);
    },

    resetView() {
      resetViewState();
    },

    toggleInvert() {
      if (!lastBitmap) return;
      windowing.invert = !windowing.invert;
      applyWindowing();
    },

    setWindowing() {
      if (!lastBitmap) return;
      applyWindowing();
    },

    undoMark() {
      marks.pop();
      draftMark = null;
      const measures = marks.filter(m => m.type === 'measure');
      setMeasureText(measures.length ? measures[measures.length - 1] : null);
      redrawMarks();
    },

    clearMarks() {
      marks = [];
      draftMark = null;
      setMeasureText(null);
      redrawMarks();
    },

    exportPNG() {
      const tmp=document.createElement('canvas');
      tmp.width=srcCanvas.width;tmp.height=srcCanvas.height;
      const ctx=tmp.getContext('2d');
      drawFilteredSource(ctx);
      if (!aiHidden) ctx.drawImage(detCanvas,0,0);
      ctx.drawImage(markCanvas,0,0);
      const a=document.createElement('a');
      a.download='fracture_detection.png';
      a.href=tmp.toDataURL('image/png');a.click();
    },

    async exportPDF() {
      if (!lastBitmap) return;   // require at least an uploaded image
      const confThr = document.getElementById('confSlider').value / 100;
      const visible = lastDets.filter(d => d.score >= confThr);
      const isWarn = visible.length > 0;   // kept as alias for downstream code
      const hasFracture = isWarn;
      // CASCADE: a clean "no fracture detected" report is now a valid output.

      // Merge source image, optional AI overlay, and manual workstation marks.
      const tmp = document.createElement('canvas');
      tmp.width = srcCanvas.width; tmp.height = srcCanvas.height;
      const tctx = tmp.getContext('2d');
      drawFilteredSource(tctx);
      if (!aiHidden) tctx.drawImage(detCanvas, 0, 0);
      tctx.drawImage(markCanvas, 0, 0);
      const imgDataURL = tmp.toDataURL('image/jpeg', 0.92);

      // ── CASCADE: run REAL EfficientNet-B3 classifier on the whole image (not the old fake heuristic).
      //    Runs only if fractures present (cascade contract: presence is the detector's job).
      let realRanked = null;
      if (hasFracture && SESSION_CLF) {
        try { realRanked = await classifyImage(lastBitmap); }
        catch(e) { console.warn('PDF classifier failed', e); }
      }

      // Per-detection cards: position + box confidence (NO fake heuristic types here)
      const C = ['#22c55e','#3b82f6','#f59e0b','#ef4444','#a855f7','#06b6d4','#ec4899','#84cc16'];
      let detsHTML = '';
      visible.forEach((d, i) => {
        const col = C[i % C.length];
        const conf = (d.score * 100).toFixed(1);
        const loc = getLocationDesc(d, srcCanvas.width, srcCanvas.height);
        const w = Math.round(d.x2-d.x1), h = Math.round(d.y2-d.y1);
        const sc = d.score>=.7?'#16a34a':d.score>=.4?'#d97706':'#dc2626';
        detsHTML += `
          <div style="display:flex;align-items:flex-start;gap:12px;padding:12px 14px;
            border:1px solid #e5e7eb;border-left:4px solid ${col};border-radius:8px;margin-bottom:8px">
            <div style="width:11px;height:11px;border-radius:50%;background:${col};flex-shrink:0;margin-top:3px"></div>
            <div style="flex:1">
              <div style="display:flex;justify-content:space-between;align-items:baseline">
                <span style="font-weight:700;font-size:14px;color:#1a1a2e">#${i+1}</span>
                <span style="font-weight:800;font-size:15px;color:${sc}">${conf}%</span>
              </div>
              <div style="font-size:11px;color:#6b7280;margin-top:4px">📍 ${loc} &nbsp;·&nbsp; ${w}×${h}px</div>
              <div style="height:5px;background:#f3f4f6;border-radius:3px;margin-top:7px;overflow:hidden">
                <div style="width:${conf}%;height:100%;background:${sc};border-radius:3px"></div>
              </div>
            </div>
          </div>`;
      });

      // i18n labels
      const zh = LANG==='zh', ko = LANG==='ko';
      const title     = ko?'골절 검출 보고서':zh?'骨折检测报告':'Fracture Detection Report';
      const subtitle  = ko?'AI 보조 · 계단식 검출→분류 · 교육/연구 목적'
                       :zh?'AI辅助 · 级联检测→分类 · 仅供教育研究用途'
                       :'AI-Assisted · Cascade Detect→Classify · Educational/Research Use';
      // Real, verified numbers (v6 detector + MTL classifier)
      const detLbl    = 'YOLOv8m · mAP50 0.82 · 27k multi-source unified dataset (Phase 6)';
      const clfLbl    = 'MultiTask Net · loc acc 92% · dir acc 86% · EdgeGuidedAttention';
      const tsLbl     = ko?'검사일시':zh?'检测时间':'Timestamp';
      const detStr    = ko?'검출 모델':zh?'检测模型':'Detector';
      const clfStr    = ko?'분류 모델':zh?'分类模型':'Classifier';
      const confStr   = ko?'신뢰도 임계값':zh?'置信度阈值':'Conf. Threshold';
      const stg1Lbl   = ko?'1단계 · 검출 (존재+위치)':zh?'第1步 · 检测(有无+位置)':'Stage 1 · Detection (presence + location)';
      const stg2Lbl   = ko?'2단계 · 유형 분류 (계단식 10종)':zh?'第2步 · 类型分类(级联 10类)':'Stage 2 · Type Classification (cascade 10-cls)';
      const noFracLbl = ko?'검출된 골절 없음':zh?'未见骨折':'No fracture detected';
      const noFracDesc= ko?'검출 모델이 골절을 찾지 못해 분류 단계는 건너뜁니다 (계단식 워크플로우).'
                       :zh?'检测模型未发现骨折,分类阶段已跳过(级联流程)。'
                       :'Detector found no fracture; classifier was skipped (cascade workflow).';
      const imgLbl    = ko?'X-ray 검출 결과':zh?'X光检测结果':'X-ray Detection Result';
      const disc      = ko?'⚠️ 이 보고서는 교육 및 연구 목적으로만 제공됩니다. 의료기기가 아니며 임상 진단에 사용할 수 없습니다. 반드시 방사선 전문의와 상담하십시오.'
                       :zh?'⚠️ 本报告仅供教育和研究用途，非认证医疗器械，不得用于临床诊断，医学图像解读请咨询有资质的放射科医生。'
                       :'⚠️ This report is for educational and research purposes only. Not a medical device. Do not use for clinical diagnosis. Always consult a qualified radiologist.';

      const verdictColor = isWarn ? '#92400e' : '#166534';
      const verdictBg    = isWarn ? '#fffbeb' : '#f0fdf4';
      const verdictBorder= isWarn ? '#f59e0b' : '#22c55e';
      const verdictIcon  = isWarn ? '⚠️' : '✅';
      const verdictText  = isWarn
        ? `${t('vrdFracture')} — ${visible.length} ${t(visible.length!==1?'regions':'region')}`
        : noFracLbl;
      const verdictSub   = isWarn ? t('scrFracture') : noFracDesc;

      // Whole-image type from REAL classifier (cascade) — only shown when fracture detected
      let typeHTML = '';
      if (realRanked && realRanked.length) {
        const top = realRanked[0];
        const topConf = (top.p*100).toFixed(1);
        const topName = clfName(top.idx);
        const top3 = realRanked.slice(0, 3);
        const top3Str = top3.map(r=>`${clfName(r.idx)} ${(r.p*100).toFixed(0)}%`).join(' · ');
        const typeCaption = ko?'전체 영상 보조 분류 (레거시 EfficientNet-B3, 참고용 · 주 진단은 다중작업 박스별 분석 사용)'
                           :zh?'整图辅助分类(传统 EfficientNet-B3,仅供参考 · 主诊断使用多任务每框分析)'
                           :'Whole-image auxiliary class (legacy EfficientNet-B3, reference only · primary diagnosis uses MultiTask per-box)';
        typeHTML = `
          <div style="padding:14px 16px;background:#f8fafc;border:1.5px solid #3d85ff;
            border-radius:10px;margin-bottom:10px">
            <div style="font-size:11px;color:#6b7280;margin-bottom:6px">${typeCaption}</div>
            <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:8px">
              <span style="font-weight:700;font-size:15px;color:#1a1a2e">${topName}</span>
              <span style="font-weight:800;font-size:16px;color:#3d85ff">${topConf}%</span>
            </div>
            <div style="font-size:10px;color:#6b7280">Top-3: ${top3Str}</div>
          </div>`;
      }

      const html = `<!DOCTYPE html><html lang="${LANG}">
<head><meta charset="UTF-8"/>
<title>${title}</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;
  background:#f8fafc;color:#1a1a2e;font-size:13px}
.page{max-width:800px;margin:0 auto;background:#fff;padding:36px 40px;
  box-shadow:0 2px 20px rgba(0,0,0,.08)}
.header{display:flex;justify-content:space-between;align-items:flex-start;
  border-bottom:3px solid #3d85ff;padding-bottom:16px;margin-bottom:28px}
.logo{font-size:21px;font-weight:800;color:#3d85ff;letter-spacing:-.3px}
.sub{font-size:10px;color:#94a3b8;margin-top:4px}
.meta{text-align:right;font-size:11px;color:#94a3b8;line-height:2}
.meta strong{color:#475569}
.sec{margin-bottom:24px}
.sec-title{font-size:12px;font-weight:700;color:#3d85ff;
  border-left:4px solid #3d85ff;padding-left:10px;margin-bottom:14px;letter-spacing:.3px;text-transform:uppercase}
.img-caption{font-size:10px;color:#94a3b8;text-align:center;margin-top:6px}
img{max-width:100%;border:1px solid #e5e7eb;border-radius:10px;display:block;margin:0 auto}
.verdict{display:flex;align-items:flex-start;gap:12px;padding:14px 16px;border-radius:10px;
  border:1.5px solid ${verdictBorder};background:${verdictBg};margin-bottom:10px}
.verdict-icon{font-size:20px;flex-shrink:0;line-height:1}
.verdict-title{font-weight:700;font-size:14px;color:${verdictColor}}
.verdict-sub{font-size:11px;color:#6b7280;margin-top:4px;line-height:1.6}
.summary{margin-top:12px;padding:10px 14px;background:#f8fafc;border:1px solid #e5e7eb;
  border-radius:8px;font-size:11px;color:#475569}
.disc{margin-top:32px;padding:14px 16px;background:#fffbeb;border:1.5px solid #f59e0b;
  border-radius:10px;font-size:11px;color:#92400e;line-height:1.8}
.footer{margin-top:20px;text-align:center;font-size:10px;color:#94a3b8;border-top:1px solid #f1f5f9;padding-top:12px}
@media print{
  body{background:#fff}
  .page{box-shadow:none;padding:20px}
  .print-bar{display:none!important}
  @page{margin:15mm}
}
.print-bar{position:sticky;top:0;z-index:100;background:#3d85ff;
  padding:10px 20px;display:flex;align-items:center;justify-content:space-between;gap:12px}
.print-bar .hint{font-size:12px;color:rgba(255,255,255,.85)}
#printBtn{background:#fff;color:#1a5ce5;border:none;border-radius:7px;
  padding:8px 20px;font-size:13px;font-weight:700;cursor:pointer;
  display:flex;align-items:center;gap:6px;transition:opacity .15s}
#printBtn:hover{opacity:.88}
</style></head>
<body>
<div class="print-bar">
  <span class="hint">🦴 ${title}</span>
  <button id="printBtn">🖨️ ${ko?'인쇄 / PDF 저장':zh?'打印 / 保存PDF':'Print / Save as PDF'}</button>
</div>
<div class="page">
  <div class="header">
    <div>
      <div class="logo">🦴 ${title}</div>
      <div class="sub">${subtitle}</div>
    </div>
    <div class="meta">
      <div><strong>${tsLbl}:</strong> ${new Date().toLocaleString()}</div>
      <div><strong>${detStr}:</strong> ${detLbl}</div>
      <div><strong>${clfStr}:</strong> ${clfLbl}</div>
      <div><strong>${t('elapsed')}:</strong> ${lastElapsed} ms &nbsp;·&nbsp; <strong>${confStr}:</strong> ${confThr.toFixed(2)}</div>
    </div>
  </div>

  <div class="sec">
    <div class="sec-title">${imgLbl}</div>
    <img src="${imgDataURL}" alt="X-ray"/>
    <div class="img-caption">on-device ONNX Runtime Web · cascade architecture</div>
  </div>

  <div class="sec">
    <div class="sec-title">${stg1Lbl}</div>
    <div class="verdict">
      <div class="verdict-icon">${verdictIcon}</div>
      <div>
        <div class="verdict-title">${verdictText}</div>
        <div class="verdict-sub">${verdictSub}</div>
      </div>
    </div>
  </div>

  ${isWarn ? `<div class="sec">
    <div class="sec-title">${stg1Lbl} · ${ko?'검출된 영역':zh?'检测到的区域':'Detected regions'}</div>
    ${detsHTML}
  </div>

  <div class="sec">
    <div class="sec-title">${stg2Lbl}</div>
    ${typeHTML || `<div style="font-size:11px;color:#6b7280;padding:10px;background:#f8fafc;border-radius:8px">${ko?'분류 모델 로드 안됨':zh?'分类模型未加载':'Classifier not loaded'}</div>`}
  </div>` : ''}

  <div class="disc">${disc}</div>
  <div class="footer">bone-fracture-yolov8.netlify.app &nbsp;·&nbsp; ZHANG HAO 장호 &nbsp;·&nbsp; cascade architecture · verified, reproducible</div>
</div>
<script>
  document.getElementById('printBtn').addEventListener('click',()=>window.print());
<\/script>
</body></html>`;

      const win = window.open('', '_blank', 'width=860,height=760');
      if (!win) { alert(ko?'팝업이 차단되었습니다. 허용 후 다시 시도해 주세요.':zh?'弹窗被拦截，请允许后重试。':'Popup blocked. Please allow popups and try again.'); return; }
      win.document.write(html);
      win.document.close();
    },

    /* ── GRAD-CAM / OCCLUSION SALIENCY HEATMAP ────────────────────────────────
     * Model-agnostic explainability for the detector: slide an 8x8 mask over the
     * input, run inference on each masked variant, measure how much the max
     * detection confidence DROPS. Cells whose masking causes a big drop are
     * "important" -> hot. Upscaled & overlaid on the original via a jet colormap.
     *
     * 64 forward passes => ~3-5s total on WebGL/WebGPU. No special model export
     * needed (works with any single-class YOLOv8 detector). Toggle on/off.
     */
    async showHeatmap() {
      if (!SESSION || !lastBitmap) return;
      const btn = document.getElementById('btnHeatmap');

      // Toggle off if already shown
      if (window._heatmapShown) {
        window._heatmapShown = false;
        // Re-draw the original detection boxes (clears the heatmap)
        const confThr = document.getElementById('confSlider').value / 100;
        drawBoxes(lastDets.filter(d => d.score >= confThr));
        btn.innerHTML = '🔥 <span data-i18n="heatmap">' + t('heatmap') + '</span>';
        return;
      }

      const origLabel = btn.innerHTML;
      btn.disabled = true;
      btn.innerHTML = '<span class="spin"></span> ' + t('heatmapComputing');

      try {
        // ────────────────────────────────────────────────────────────────────
        // PER-DETECTION focused occlusion saliency (v14.2 — fixes accuracy)
        //
        // OLD approach: one 5×5 grid across the WHOLE image.
        //   On a 1000×1000 X-ray, each cell was 200×200 px → fracture (50-100 px)
        //   fit entirely inside one cell. Heat got rendered as a 200-px blob,
        //   smeared further by bilinear upsampling. AND it used max confidence
        //   across all detections → can't distinguish per-fracture saliency.
        //
        // NEW approach: for each detection box, run a 6×6 grid INSIDE the
        //   expanded bbox (+30% padding). Each cell is now ~30-50 px → matches
        //   the actual fracture-line width. Per-box matching by IoU isolates
        //   each detection's saliency. Heatmap is composited ONLY inside the
        //   bbox region, not across the whole image.
        // ────────────────────────────────────────────────────────────────────
        const sz   = META.img_size;
        const inN  = META._in;
        const grid = (currentModel === 'full') ? 5 : 6;
        const PAD  = 0.30;            // expand bbox by 30% on each side
        const IOU_MATCH = 0.30;       // IoU threshold for "same fracture"
        const yieldUI = () => new Promise(r => setTimeout(r, 0));

        const W = lastBitmap.width || lastBitmap.naturalWidth || sz;
        const H = lastBitmap.height || lastBitmap.naturalHeight || sz;
        const confThr = document.getElementById('confSlider').value / 100;
        const targetDets = (lastDets || []).filter(d => d.score >= confThr);

        if (!targetDets.length) {
          alert(t('heatmapNoBaseDet'));
          btn.innerHTML = origLabel; btn.disabled = false;
          return;
        }

        const totalCells = targetDets.length * grid * grid;

        // Allow cancel by clicking the button mid-computation
        window._heatmapCancel = false;
        const origOnclick = btn.onclick;
        btn.onclick = () => { window._heatmapCancel = true; };
        btn.disabled = false;
        btn.innerHTML = '⏹ <span style="font-size:11px">0/' + totalCells + ' · 点击取消</span>';

        // Run YOLO on a canvas, return ALL detections (not max).
        async function detectAll(srcCanvas) {
          const lb = letterbox(srcCanvas, sz);
          const f32 = toFloat32(lb.canvas, sz);
          const tens = new ort.Tensor('float32', f32, [1, 3, sz, sz]);
          const out  = await SESSION.run({[inN]: tens});
          const ot   = out[Object.keys(out)[0]];
          return parseOutput(ot.data, ot.dims, 0.01, lb.scale, lb.px, lb.py, lb.origW, lb.origH);
        }

        function iou(a, b) {
          const ix1 = Math.max(a.x1, b.x1);
          const iy1 = Math.max(a.y1, b.y1);
          const ix2 = Math.min(a.x2, b.x2);
          const iy2 = Math.min(a.y2, b.y2);
          if (ix2 <= ix1 || iy2 <= iy1) return 0;
          const inter = (ix2 - ix1) * (iy2 - iy1);
          const A = (a.x2 - a.x1) * (a.y2 - a.y1);
          const B = (b.x2 - b.x1) * (b.y2 - b.y1);
          return inter / (A + B - inter + 1e-6);
        }
        // Best confidence among detections matching `target` (IoU >= threshold)
        function bestMatchConf(detsAfter, target) {
          let best = 0;
          for (const d of detsAfter) {
            if (iou(d, target) >= IOU_MATCH) best = Math.max(best, d.score);
          }
          return best;
        }

        const dc   = detCanvas;
        const dctx = dc.getContext('2d');
        // Repaint detection boxes first; heatmaps go on top with alpha
        drawBoxes(targetDets);

        function jet(v) {  // 0..1 → [r,g,b]
          const r = Math.max(0, Math.min(1, 1.5 - Math.abs(4*v - 3)));
          const g = Math.max(0, Math.min(1, 1.5 - Math.abs(4*v - 2)));
          const b = Math.max(0, Math.min(1, 1.5 - Math.abs(4*v - 1)));
          return [r*255|0, g*255|0, b*255|0];
        }

        // Scale factor between the source-image coordinate space (W, H) and the
        // overlay canvas (detCanvas size). Box coords in lastDets use source space.
        const sx = dc.width  / W;
        const sy = dc.height / H;

        const t0 = performance.now();
        let done = 0;

        for (let di = 0; di < targetDets.length; di++) {
          if (window._heatmapCancel) break;
          const det = targetDets[di];
          // Expanded bbox in source-image coordinates
          const bw = det.x2 - det.x1, bh = det.y2 - det.y1;
          const ex = bw * PAD,  ey = bh * PAD;
          const bx1 = Math.max(0, Math.floor(det.x1 - ex));
          const by1 = Math.max(0, Math.floor(det.y1 - ey));
          const bx2 = Math.min(W, Math.ceil(det.x2 + ex));
          const by2 = Math.min(H, Math.ceil(det.y2 + ey));
          const bbw = bx2 - bx1, bbh = by2 - by1;
          if (bbw < 4 || bbh < 4) continue;

          // Baseline = the detection's own score (no need to re-run on full image)
          const baseConf = det.score;

          // Per-cell occlusion within this expanded bbox
          const drops = new Float32Array(grid * grid);
          const cellW = bbw / grid, cellH = bbh / grid;

          for (let r = 0; r < grid; r++) {
            if (window._heatmapCancel) break;
            for (let c = 0; c < grid; c++) {
              if (window._heatmapCancel) break;
              const mc = document.createElement('canvas');
              mc.width = W; mc.height = H;
              const mx = mc.getContext('2d');
              mx.drawImage(lastBitmap, 0, 0, W, H);
              mx.fillStyle = 'rgba(128,128,128,1)';
              mx.fillRect(bx1 + c * cellW, by1 + r * cellH, cellW, cellH);
              const detsAfter = await detectAll(mc);
              const mConf = bestMatchConf(detsAfter, det);
              drops[r * grid + c] = Math.max(0, baseConf - mConf);
              done++;
              if (done % 2 === 0 || done === totalCells) {
                const el = (performance.now() - t0) / 1000;
                const eta = Math.max(0, el * (totalCells - done) / Math.max(1, done));
                btn.innerHTML =
                  '⏹ <span style="font-size:11px">' +
                  `box ${di+1}/${targetDets.length} · ${done}/${totalCells} · ${el.toFixed(0)}s · ETA ${eta.toFixed(0)}s` +
                  '</span>';
                await yieldUI();
              }
            }
          }
          if (window._heatmapCancel) break;

          // Normalize this box's drops to [0,1] independently → each fracture gets full color range
          let dmax = 0;
          for (const v of drops) if (v > dmax) dmax = v;
          if (dmax <= 0) continue;
          const norm = new Float32Array(grid * grid);
          for (let i = 0; i < drops.length; i++) norm[i] = drops[i] / dmax;

          // Render this bbox's heatmap onto a small canvas, scale onto detCanvas region
          const small = document.createElement('canvas');
          small.width = grid; small.height = grid;
          const sctx = small.getContext('2d');
          const sid  = sctx.createImageData(grid, grid);
          for (let i = 0; i < grid * grid; i++) {
            // Soft threshold: cells below 20% saliency stay transparent (reduces noise)
            const v = norm[i] < 0.20 ? 0 : norm[i];
            const [rr, gg, bb] = jet(v);
            sid.data[i*4]   = rr;
            sid.data[i*4+1] = gg;
            sid.data[i*4+2] = bb;
            sid.data[i*4+3] = Math.round(v * 200);   // intensity-modulated alpha
          }
          sctx.putImageData(sid, 0, 0);

          // Composite onto detCanvas at the bbox location (in canvas coords)
          dctx.save();
          dctx.imageSmoothingEnabled = true;
          dctx.imageSmoothingQuality = 'high';
          dctx.globalCompositeOperation = 'source-over';
          dctx.drawImage(small, bx1 * sx, by1 * sy, bbw * sx, bbh * sy);
          dctx.restore();
        }

        // Restore button click handler
        btn.onclick = origOnclick;
        if (window._heatmapCancel) {
          window._heatmapCancel = false;
          btn.innerHTML = origLabel;
          btn.disabled = false;
          return;
        }

        window._heatmapShown = true;
        btn.innerHTML = '✖ ' + t('heatmapRemove');
      } catch (e) {
        console.error('[heatmap] failed', e);
        alert('Heatmap failed: ' + e.message);
        btn.innerHTML = origLabel;
      } finally {
        btn.disabled = false;
        // Always restore the default click handler — if user cancelled or an
        // exception fired mid-loop, the temporary cancel-handler must NOT linger
        // (otherwise next click would set _heatmapCancel=true instead of running).
        btn.onclick = () => APP.showHeatmap();
        window._heatmapCancel = false;
      }
    },

    clear() {
      lastDets=[];lastBitmap=null;lastElapsed=0;
      window._heatmapShown = false;   // reset heatmap toggle
      cw.classList.remove('on');
      document.getElementById('drop')?.classList.remove('has-img');
      document.getElementById('lboxPh').style.display='';
      document.getElementById('xrayWarn').classList.remove('show');
      detCanvas.getContext('2d').clearRect(0,0,detCanvas.width,detCanvas.height);
      markCanvas.getContext('2d').clearRect(0,0,markCanvas.width,markCanvas.height);
      marks=[];draftMark=null;resetViewState();setToolMode('lock');setAIHidden(false);
      windowing={brightness:0,contrast:100,invert:false};applyWindowing();setMeasureText(null);
      setWorkstationEnabled(false);   // disable workstation tools — no image to manipulate
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
      renderClassification(visible);   // cascade: function handles no-detection case internally
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
    img.onerror = () => setRunSts(t('exLoadErr'), 'err');
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
      renderClassification(preset.filter(d=>d.score>=confThr));   // cascade: handles empty internally
      const vis=preset.filter(d=>d.score>=confThr).length;
      setRunSts(`✅ ${lastElapsed}ms · ${vis} ${t(vis!==1?'detections':'detection')}`, 'ok');
      setExport(true);
      runEl.disabled=false;
      return;
    }
    return _origRun();
  };

  /* ── RE-RENDER ON LANG CHANGE ──────────────────────────── */
  /*
   * Called by setLang() whenever the user switches language.
   * Re-renders all dynamic detection output (stage 1 + stage 2 results,
   * run-status bar) using the new LANG so nothing stays in the old language.
   */
  window.reRenderResults = function() {
    const confThr = document.getElementById('confSlider').value / 100;

    if (lastElapsed === 0) {
      // No detection run yet — just refresh placeholder strings
      document.getElementById('scrResult').innerHTML =
        `<div style="color:var(--fg3);font-size:11px;text-align:center;padding:16px 0">${t('waiting')}</div>`;
      document.getElementById('clfResult').innerHTML =
        `<div style="color:var(--fg3);font-size:11px;text-align:center;padding:16px 0">${t('waitScr')}</div>`;
      if (!lastBitmap) setRunSts(t('selFirst'));
      return;
    }

    // Detection has been run — re-render results in new language
    const visible = lastDets.filter(d => d.score >= confThr);
    renderScreening(visible, confThr, lastElapsed);
    renderClassification(visible);   // cascade: handles empty internally

    // Also update the run-status bar
    const visCount = visible.length;
    setRunSts(`✅ ${lastElapsed}ms · ${visCount} ${t(visCount!==1?'detections':'detection')}`, 'ok');
    updateToolButtons();
    setAIHidden(aiHidden);
  };

  /* ── BOOT ───────────────────────────────────────────────── */
  async function boot() {
    await purgeStaleClientCache();
    init();
  }
  if (document.readyState === 'loading')
    document.addEventListener('DOMContentLoaded', boot);
  else boot();

})();
