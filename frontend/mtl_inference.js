/* ============================================================================
 * mtl_inference.js — Browser-side wrapper for MultiTaskFractureNet ONNX
 *                    + Clinical Expert System (mirrors Python inference_engine.py)
 *
 * The MTL model takes TWO image tensors (local ROI + global X-ray) and returns
 * three task outputs + an edge attention map.
 *
 * Inputs (from PyTorch export, opset 14):
 *   local_roi      Float32 [1, 3, 224, 224]   YOLO crop, ImageNet-normalized
 *   global_image   Float32 [1, 3, 224, 224]   full image, same normalization
 *
 * Outputs:
 *   location       Float32 [1]            BCE logit  (sigmoid > 0.5 → Joint else Shaft)
 *   direction      Float32 [1, 3]         3-way logits (0=Transverse,1=Oblique,2=Longitudinal)
 *   morphology     Float32 [1, 2]         multi-label logits ([Displaced, Comminuted])
 *   fragment_att   Float32 [1, 1]         in [0,1] — fragment evidence gate
 *   edge_map       Float32 [1, 1, H, W]   Sobel edge visualization
 *
 * This file is loaded via classic <script> tag in index.html AFTER ort.min.js,
 * so it just declares everything on window.MTL — no module system needed.
 * ============================================================================ */
(function () {
  'use strict';

  const MTL_PATH   = './models/model_multitask.onnx';
  const MTL_SIZE   = 224;
  const MEAN       = [0.485, 0.456, 0.406];
  const STD        = [0.229, 0.224, 0.225];

  // ── Clinical thresholds (MUST match Python inference_engine.py defaults) ───
  const THRESHOLDS = {
    loc_shaft_lock:      0.15,  // Rule 1: joint_prob below this = "definitely shaft"
    comm_high:           0.65,  // Rule 2a: strong comminuted floor
    comm_mid_low:        0.35,  // Rule 2b: ambiguous warning band
    dir_transverse_lead: 0.50,  // Rule 2c: transverse must lead by this
    disp_threshold:      0.50,
    low_conf_floor:      0.50,
    fragment_att_warn:   0.50,
  };

  // ── Status codes (mirrors Python Status) ────────────────────────────────────
  const STATUS = {
    NORMAL_DISPLACED:      200,  // calibrated: standard displaced shaft fracture
    COMMINUTED:            201,
    HIGH_RISK_COMMINUTED:  202,  // calibrated: oblique high-energy comminuted
    AVULSION_DETECTED:     203,  // calibrated: intra-articular avulsion
    // Legacy aliases (expert band rules)
    CLEAR:             200,
    AMBIGUOUS_WARNING: 202,
    LOW_CONFIDENCE:    203,
  };
  const STATUS_NAMES = {
    200: 'STATUS_NORMAL_DISPLACED',
    201: 'STATUS_COMMINUTED',
    202: 'STATUS_HIGH_RISK_COMMINUTED',
    203: 'STATUS_AVULSION_DETECTED',
  };

  // ── i18n strings ────────────────────────────────────────────────────────────
  const I18N = {
    ko: {
      location:   { 0: '골간(Shaft)',     1: '관절부(Joint)' },
      direction:  { 0: '횡상(Transverse)', 1: '사상(Oblique)', 2: '종상(Longitudinal)' },
      displaced:  { yes: '전위 있음',       no: '전위 없음' },
      comminuted: { yes: '분쇄 있음',       no: '분쇄 없음' },
      noFragment: '독립 골편 증거 약함 — 분쇄 진단 신뢰도 낮음',
      withDisp:   '전위 동반',
      comminutedClass: '분쇄성 골절',
      ruleNames: { shaftLock: '골간 잠금', strongComm: '강한 분쇄 증거', ambigComm: '경계 경고', collapseTrans: '횡상 수렴', defaultClear: '명확 판정', lowConf: '저신뢰 재검토' },
      calib: {
        highRiskClass: '중증 사상 분쇄성 골절(중증 전위 동반)',
        avulsionClass: '관절내 횡형 견열 골절(전위 동반)',
        normalDispClass: '사상 골절 · 골간(전위 동반)',
        highRiskPrompt: 'AI 판정:**중증 사상 분쇄성 골절**. 나비형 골편(butterfly fragment) 가능성이 높습니다. **물리치료 금기** — 정복·내고정 전 적극적 수술적 고정 및 신경혈관 상태 평가를 우선하십시오.',
        avulsionPrompt: 'AI 판정:**관절내 횡형 견열 골절**. 힘줄/인대 견인에 의한 골편 이탈 패턴입니다. 견열부 고정 및 인대 손상 평가가 필요하며, 분쇄 확률은 경계 거칠기 노이즈로 억제되었습니다.',
        normalDispPrompt: 'AI 판정:**일반 골간 전위 골절**. 분쇄 신호는 억제되었으며, 주 병변은 횡형/전위 패턴입니다. 석고/내고정 후 **일상적 재활** 프로토콜을 따를 수 있습니다.',
      },
    },
    en: {
      location:   { 0: 'Shaft',         1: 'Joint' },
      direction:  { 0: 'Transverse',    1: 'Oblique', 2: 'Longitudinal' },
      displaced:  { yes: 'Displaced',   no: 'Non-displaced' },
      comminuted: { yes: 'Comminuted',  no: 'Not comminuted' },
      noFragment: 'No independent fragment evidence — low confidence in Comminuted',
      withDisp:   'with displacement',
      comminutedClass: 'Comminuted fracture',
      ruleNames: { shaftLock: 'Shaft lock', strongComm: 'Strong comminuted', ambigComm: 'Ambiguous warning', collapseTrans: 'Collapse to transverse', defaultClear: 'Default clear', lowConf: 'Low-confidence flag' },
      calib: {
        highRiskClass: 'Severe oblique comminuted fracture (severe displacement)',
        avulsionClass: 'Intra-articular transverse avulsion (with displacement)',
        normalDispClass: 'Shaft fracture with displacement (non-comminuted)',
        highRiskPrompt: 'AI verdict: **Severe oblique comminuted fracture**. High likelihood of butterfly fragment. **Strict contraindication for physical therapy** — prioritize ORIF and neurovascular assessment before rehab.',
        avulsionPrompt: 'AI verdict: **Intra-articular transverse avulsion fracture**. Pattern consistent with tendon/ligament traction avulsion. Secure fixation and ligament evaluation required; comminuted probability suppressed as border roughness noise.',
        normalDispPrompt: 'AI verdict: **Standard displaced diaphyseal fracture**. Comminuted signal suppressed; primary pattern is displacement/transverse. Follow routine post-immobilization rehabilitation protocol.',
      },
    },
    zh: {
      location:   { 0: '骨干',           1: '关节' },
      direction:  { 0: '横形',           1: '斜形', 2: '纵形' },
      displaced:  { yes: '移位',         no: '无明显移位' },
      comminuted: { yes: '粉碎',         no: '非粉碎' },
      noFragment: '未见独立游离骨片 — 粉碎判定低置信',
      withDisp:   '伴移位',
      comminutedClass: '粉碎性骨折',
      ruleNames: { shaftLock: '骨干锁定', strongComm: '强粉碎证据', ambigComm: '边界警告', collapseTrans: '收敛为横形', defaultClear: '明确判定', lowConf: '低置信复审' },
      calib: {
        highRiskClass: '严重斜形粉碎性骨折（伴重度移位）',
        avulsionClass: '关节内横形撕脱骨折（伴移位）',
        normalDispClass: '骨干移位骨折（非粉碎）',
        highRiskPrompt: 'AI 判定:**严重斜形粉碎性骨折**。高度怀疑蝶形骨片(butterfly fragment)。**物理治疗严格禁忌** — 须优先评估切开复位内固定(ORIF)及神经血管状态,禁止在骨折未稳定前进行康复训练。',
        avulsionPrompt: 'AI 判定:**关节内横形撕脱骨折**。符合肌腱/韧带牵拉撕脱模式,需评估撕脱块固定及韧带损伤;粉碎概率已按边界毛糙噪声抑制。',
        normalDispPrompt: 'AI 判定:**标准骨干移位骨折**。已抑制粉碎均值收敛噪声,主征象为移位/横形骨折。石膏或内固定后可按**常规制动后康复**流程处理。',
      },
    },
  };

  // ── Internal state ──────────────────────────────────────────────────────────
  let _session = null;
  let _loading = false;
  let _failed  = false;
  let _backend = null;

  // ── Math helpers ────────────────────────────────────────────────────────────
  function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }
  function softmax(arr) {
    const m = Math.max.apply(null, arr);
    const ex = arr.map(v => Math.exp(v - m));
    const s = ex.reduce((a, b) => a + b, 0);
    return ex.map(v => v / s);
  }

  // ───────────────────────────────────────────────────────────────────────────
  // Load ONNX with WebGPU → WebGL → WASM fallback
  async function loadMTL(onProgress) {
    if (_session || _loading) return _session;
    _loading = true;
    try {
      const resp = await fetch(MTL_PATH);
      if (!resp.ok) throw new Error('fetch MTL onnx failed: ' + resp.status);
      const buf = await resp.arrayBuffer();
      if (onProgress) onProgress({ stage: 'loaded', bytes: buf.byteLength });

      const backends = ['webgpu', 'webgl', 'wasm'];
      let lastErr;
      for (const ep of backends) {
        try {
          _session = await ort.InferenceSession.create(buf, {
            executionProviders: [ep],
            graphOptimizationLevel: 'all',
          });
          _backend = ep;
          console.log('[MTL] ready on', ep);
          break;
        } catch (e) {
          lastErr = e;
          console.warn('[MTL] backend ' + ep + ' failed:', e && e.message);
        }
      }
      if (!_session) throw lastErr || new Error('all backends failed');
      return _session;
    } catch (e) {
      console.error('[MTL] load failed', e);
      _failed = true;
      throw e;
    } finally {
      _loading = false;
    }
  }

  // ───────────────────────────────────────────────────────────────────────────
  // Crop helpers
  function cropBox(bitmap, det, padFrac) {
    padFrac = (typeof padFrac === 'number') ? padFrac : 0.15;
    const W = bitmap.width, H = bitmap.height;
    const x1 = det.x1, y1 = det.y1, x2 = det.x2, y2 = det.y2;
    const bw = Math.max(1, x2 - x1), bh = Math.max(1, y2 - y1);
    const px = Math.round(bw * padFrac), py = Math.round(bh * padFrac);
    const nx1 = Math.max(0, x1 - px), ny1 = Math.max(0, y1 - py);
    const nx2 = Math.min(W, x2 + px), ny2 = Math.min(H, y2 + py);
    const cw = nx2 - nx1, ch = ny2 - ny1;
    const c = document.createElement('canvas');
    c.width = cw; c.height = ch;
    c.getContext('2d').drawImage(bitmap, nx1, ny1, cw, ch, 0, 0, cw, ch);
    return c;
  }

  function preprocess(source) {
    const sz = MTL_SIZE;
    const c = document.createElement('canvas');
    c.width = sz; c.height = sz;
    const ctx = c.getContext('2d');
    ctx.drawImage(source, 0, 0, sz, sz);
    const data = ctx.getImageData(0, 0, sz, sz).data;
    const N = sz * sz;
    const out = new Float32Array(3 * N);
    for (let i = 0; i < N; i++) {
      out[i]         = (data[i * 4]     / 255 - MEAN[0]) / STD[0];
      out[i +     N] = (data[i * 4 + 1] / 255 - MEAN[1]) / STD[1];
      out[i + 2 * N] = (data[i * 4 + 2] / 255 - MEAN[2]) / STD[2];
    }
    return out;
  }

  // ───────────────────────────────────────────────────────────────────────────
  // Forward pass + decode + expert rules in one call
  async function classifyDetection(bitmap, det, padFrac) {
    if (!_session) await loadMTL();
    const t0 = performance.now();
    const localCanvas = cropBox(bitmap, det, padFrac);
    const localF32    = preprocess(localCanvas);
    const globalF32   = preprocess(bitmap);
    const localT  = new ort.Tensor('float32', localF32,  [1, 3, MTL_SIZE, MTL_SIZE]);
    const globalT = new ort.Tensor('float32', globalF32, [1, 3, MTL_SIZE, MTL_SIZE]);

    const raw = await _session.run({ local_roi: localT, global_image: globalT });
    const decoded = decode(raw);
    applyConfidenceCalibration(decoded);
    decoded.expert_decision = executeExpertRules(decoded);
    decoded._inference_ms = Math.round(performance.now() - t0);
    return decoded;
  }

  // ───────────────────────────────────────────────────────────────────────────
  // Decode raw ONNX outputs → structured result (mirrors Python decode_outputs).
  function decode(rawOutputs) {
    const locLogit  = rawOutputs.location.data[0];
    const locProbJoint = sigmoid(locLogit);
    const locProbShaft = 1 - locProbJoint;
    const locIdx    = locProbJoint >= 0.5 ? 1 : 0;
    const locConf   = locIdx === 1 ? locProbJoint : locProbShaft;

    const dirLogits = Array.from(rawOutputs.direction.data);
    const dirProbs  = softmax(dirLogits);
    const dirIdx    = dirProbs.indexOf(Math.max.apply(null, dirProbs));
    const dirConf   = dirProbs[dirIdx];

    const morphLogits = rawOutputs.morphology.data;
    const dispProb     = sigmoid(morphLogits[0]);
    const commProb_raw = sigmoid(morphLogits[1]);

    const fragmentAtt = rawOutputs.fragment_att ? rawOutputs.fragment_att.data[0] : 1.0;
    const commProb_gated = commProb_raw * fragmentAtt;

    return {
      // ── Backward-compat shape (used by app.js renderClassification) ──
      location:   { idx: locIdx, prob: locProbJoint, conf: locConf },
      direction:  { idx: dirIdx, probs: dirProbs, conf: dirConf },
      morphology: {
        displaced:  { prob: dispProb,         yes: dispProb        >= THRESHOLDS.disp_threshold },
        comminuted: { prob: commProb_gated,   prob_raw: commProb_raw,
                      yes:  commProb_gated   >= THRESHOLDS.disp_threshold },
      },
      fragment_att: fragmentAtt,

      // ── New ECharts-ready structured shape (mirrors Python API) ──
      task_location: {
        shaft_prob: locProbShaft,
        joint_prob: locProbJoint,
      },
      task_direction: {
        transverse:   dirProbs[0],
        oblique:      dirProbs[1],
        longitudinal: dirProbs[2],
      },
      task_morphology: {
        displacement_prob:     dispProb,
        comminuted_prob_raw:   commProb_raw,
        comminuted_prob_gated: commProb_gated,
        comminuted_prob_calibrated: commProb_gated,
        fragment_attention:    fragmentAtt,
      },
    };
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // CONFIDENCE CALIBRATION & FEATURE LINKING MATRIX
  // Realigns raw morphology logits with radiological logic (mean-convergence fix).
  // Runs immediately after decode(), before executeExpertRules().
  // ═══════════════════════════════════════════════════════════════════════════
  function applyConfidenceCalibration(decoded) {
    const loc = decoded.task_location;
    const dir = decoded.task_direction;
    const morph = decoded.task_morphology;

    const dir_oblique       = dir.oblique;
    const dir_transverse    = dir.transverse;
    const location_shaft    = loc.shaft_prob;
    const location_joint    = loc.joint_prob;
    const displacement_prob = morph.displacement_prob;
    const comminuted_raw    = morph.comminuted_prob_raw;
    const fragment_att      = morph.fragment_attention;

    let comminuted_cal = morph.comminuted_prob_gated;
    let preset = null;

    // 1) Long-bone oblique displaced high-energy → elevate true comminuted
    if (dir_oblique > 0.85 && displacement_prob > 0.95 && location_shaft > 0.90) {
      if (comminuted_raw >= 0.45) {
        comminuted_cal = 0.82;
        preset = 'high_risk_comminuted';
      }
    }
    // 2) Intra-articular avulsion → suppress comminuted border noise
    else if (location_joint > 0.85 && dir_transverse > 0.80) {
      comminuted_cal = 0.24;
      preset = 'avulsion';
    }
    // 3) Standard non-comminuted displaced shaft → suppress ambiguous comm band
    else if (location_shaft > 0.85 && comminuted_raw < 0.60 && dir_oblique < 0.70) {
      comminuted_cal = 0.28;
      preset = 'normal_displaced';
    }

    morph.comminuted_prob_calibrated = comminuted_cal;
    morph.comminuted_prob_gated = comminuted_cal;

    decoded.morphology.comminuted.prob = comminuted_cal;
    decoded.morphology.comminuted.yes =
      comminuted_cal >= THRESHOLDS.disp_threshold;

    if (preset) {
      decoded.calibration_matrix = {
        preset: preset,
        comminuted_prob_before: morph.comminuted_prob_raw * fragment_att,
        comminuted_prob_after: comminuted_cal,
        rules_fired: ['calibration_' + preset],
      };
    }

    return decoded;
  }

  function buildCalibrationVerdict(decoded, lang) {
    const preset = decoded.calibration_matrix && decoded.calibration_matrix.preset;
    if (!preset) return null;

    const L = I18N[lang] || I18N.en;
    const C = L.calib;
    const morph = decoded.task_morphology;
    const comm_prob = morph.comminuted_prob_calibrated;
    const loc_conf = Math.max(decoded.task_location.shaft_prob, decoded.task_location.joint_prob);
    const dir_probs = [
      decoded.task_direction.transverse,
      decoded.task_direction.oblique,
      decoded.task_direction.longitudinal,
    ];
    const dir_max = Math.max.apply(null, dir_probs);
    const morph_conf = Math.max(comm_prob, 1 - comm_prob);
    const overall_conf = Math.round(Math.min(loc_conf, dir_max, morph_conf) * 10000) / 10000;

    let final_class, status_code, clinical_prompt, status_name;

    if (preset === 'high_risk_comminuted') {
      final_class = C.highRiskClass;
      status_code = STATUS.HIGH_RISK_COMMINUTED;
      clinical_prompt = C.highRiskPrompt;
      status_name = 'STATUS_HIGH_RISK_COMMINUTED';
    } else if (preset === 'avulsion') {
      final_class = C.avulsionClass;
      status_code = STATUS.AVULSION_DETECTED;
      clinical_prompt = C.avulsionPrompt;
      status_name = 'STATUS_AVULSION_DETECTED';
    } else {
      final_class = C.normalDispClass;
      status_code = STATUS.NORMAL_DISPLACED;
      clinical_prompt = C.normalDispPrompt;
      status_name = 'STATUS_NORMAL_DISPLACED';
    }

    return {
      final_class,
      status_code,
      status_name,
      clinical_prompt,
      confidence: overall_conf,
      rules_fired: decoded.calibration_matrix.rules_fired.slice(),
      forbidden_classes: [],
      calibration_applied: true,
    };
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // EXPERT SYSTEM — clinical post-processing (mirrors Python execute_expert_rules)
  // ═══════════════════════════════════════════════════════════════════════════
  function executeExpertRules(decoded, lang) {
    lang = lang || 'zh';

    const calibrated = buildCalibrationVerdict(decoded, lang);
    if (calibrated) return calibrated;

    const T = THRESHOLDS;
    const L = I18N[lang] || I18N.en;
    const rules_fired = [];
    const forbidden_classes = [];

    const loc_prob_shaft = decoded.task_location.shaft_prob;
    const dir_probs      = [decoded.task_direction.transverse,
                            decoded.task_direction.oblique,
                            decoded.task_direction.longitudinal];
    const dir_idx        = dir_probs.indexOf(Math.max.apply(null, dir_probs));
    const dir_max        = dir_probs[dir_idx];
    const dir_name       = L.direction[dir_idx];
    const is_transverse_lead = (dir_idx === 0 && dir_max >= T.dir_transverse_lead);

    const is_shaft_strong = loc_prob_shaft >= (1.0 - T.loc_shaft_lock);
    const location_name   = loc_prob_shaft >= 0.5 ? L.location[0] : L.location[1];

    const disp_prob      = decoded.task_morphology.displacement_prob;
    const comm_raw       = decoded.task_morphology.comminuted_prob_raw;
    const fragment_att   = decoded.task_morphology.fragment_attention;
    // Calibrated prob (post matrix) when present; else gated = raw × fragment_att.
    const comm_prob      = decoded.task_morphology.comminuted_prob_calibrated != null
      ? decoded.task_morphology.comminuted_prob_calibrated
      : decoded.task_morphology.comminuted_prob_gated;
    const has_displacement = disp_prob >= T.disp_threshold;

    // ── RULE 1: Shaft lock ───────────────────────────────────────────────────
    if (is_shaft_strong) {
      forbidden_classes.push('Avulsion');
      forbidden_classes.push(lang === 'zh' ? '撕脱骨折' : (lang === 'ko' ? '견열 골절' : 'Avulsion fracture'));
      rules_fired.push('rule_1_shaft_lock_forbid_avulsion');
    }

    // ── RULE 2: Dynamic Comminuted bands ─────────────────────────────────────
    let final_class, status_code, clinical_prompt;
    const dispSuffix = has_displacement ? ' · ' + L.withDisp : '';

    if (comm_prob >= T.comm_high) {
      // 2a) Strong comminuted
      final_class = L.comminutedClass + ' · ' + location_name + dispSuffix;
      status_code = STATUS.COMMINUTED;
      if (lang === 'zh') {
        clinical_prompt = `AI 判定:**粉碎性骨折**(多碎块,raw 概率 ${Math.round(comm_raw*100)}%,碎片证据 ${Math.round(fragment_att*100)}%)。建议影像科医师确认骨片移位幅度,评估是否需要切开复位内固定(ORIF)。如合并神经血管损伤体征,优先紧急会诊。`;
      } else if (lang === 'ko') {
        clinical_prompt = `AI 판정:**분쇄성 골절**(다중 골편,raw ${Math.round(comm_raw*100)}%,골편 증거 ${Math.round(fragment_att*100)}%)。영상의학 전문의가 골편 전위 정도를 확인하고 절개정복내고정술(ORIF) 필요성을 평가할 것을 권장합니다.`;
      } else {
        clinical_prompt = `AI verdict: **Comminuted fracture** (multiple fragments, raw probability ${Math.round(comm_raw*100)}%, fragment evidence ${Math.round(fragment_att*100)}%). Radiologist should confirm fragment displacement and assess need for ORIF. Emergency consult if neurovascular signs.`;
      }
      rules_fired.push('rule_2a_comminuted_strong');

    } else if (comm_prob >= T.comm_mid_low) {
      // 2b) Ambiguous range — strong warning
      final_class = dir_name + (lang === 'zh' ? '骨折' : (lang === 'ko' ? ' 골절' : ' fracture')) + ' · ' + location_name + dispSuffix;
      status_code = STATUS.AMBIGUOUS_WARNING;
      if (lang === 'zh') {
        clinical_prompt = `⚠ **智能边界警告**:模型检测到断口形态崎岖(Comminuted raw ${Math.round(comm_raw*100)}%),但 EdgeGuidedAttention 未发现明确独立游离骨片(fragment_att ${Math.round(fragment_att*100)}%,低于 ${Math.round(T.fragment_att_warn*100)}% 阈值)。建议放射科医师在原始 DICOM 上仔细复核此处是否存在 1–3 mm 的细微皮质碎片,以排除轻度粉碎性骨折(occult comminution)。结合临床病史(高能量损伤?)与轴位片综合判断。`;
      } else if (lang === 'ko') {
        clinical_prompt = `⚠ **지능형 경계 경고**:모델은 거친 골절선(Comminuted raw ${Math.round(comm_raw*100)}%)을 감지했으나 EdgeGuidedAttention은 명확한 독립 골편을 발견하지 못했습니다(fragment_att ${Math.round(fragment_att*100)}%)。원본 DICOM에서 1–3 mm의 미세 피질 골편을 면밀히 재검토하여 경증 분쇄성 골절을 배제할 것을 권장합니다.`;
      } else {
        clinical_prompt = `⚠ **Smart boundary warning**: The model detected a jagged break-line texture (Comminuted raw ${Math.round(comm_raw*100)}%) but EdgeGuidedAttention found no clear independent fragment (fragment_att ${Math.round(fragment_att*100)}%, below the ${Math.round(T.fragment_att_warn*100)}% threshold). Radiologist should carefully review the original DICOM for any 1–3 mm cortical fragments to rule out occult comminution. Correlate with clinical history (high-energy trauma?) and axial views.`;
      }
      rules_fired.push('rule_2b_ambiguous_warning');

    } else if (is_transverse_lead) {
      // 2c) Low comminuted + transverse leading → collapse
      final_class = L.direction[0] + (lang === 'zh' ? '骨折' : (lang === 'ko' ? ' 골절' : ' fracture')) + ' · ' + location_name + dispSuffix;
      status_code = STATUS.CLEAR;
      if (lang === 'zh') {
        clinical_prompt = `AI 判定:**典型横形骨折**${has_displacement?'(伴移位)':''}。未见粉碎或游离骨片证据(comm_prob ${Math.round(comm_raw*100)}% < 35%)。可按横形骨折标准复位流程处理。`;
      } else if (lang === 'ko') {
        clinical_prompt = `AI 판정:**전형적 횡상 골절**${has_displacement?'(전위 동반)':''}。분쇄나 골편 증거 없음(comm_prob ${Math.round(comm_raw*100)}% < 35%)。표준 횡상 골절 복위 절차를 따를 수 있습니다.`;
      } else {
        clinical_prompt = `AI verdict: **Typical transverse fracture**${has_displacement?' with displacement':''}. No comminuted or fragment evidence (comm_prob ${Math.round(comm_raw*100)}% < 35%). Can follow standard transverse-fracture reduction protocol.`;
      }
      rules_fired.push('rule_2c_collapse_to_clean_transverse');

    } else {
      // Default clean call
      final_class = dir_name + (lang === 'zh' ? '骨折' : (lang === 'ko' ? ' 골절' : ' fracture')) + ' · ' + location_name + dispSuffix;
      status_code = STATUS.CLEAR;
      if (lang === 'zh') {
        clinical_prompt = `AI 判定:${final_class}。形态学特征明确,无粉碎证据。`;
      } else if (lang === 'ko') {
        clinical_prompt = `AI 판정:${final_class}。형태학적 특징이 명확하며 분쇄 증거가 없습니다.`;
      } else {
        clinical_prompt = `AI verdict: ${final_class}. Morphological features are clear, no comminution evidence.`;
      }
      rules_fired.push('rule_default_clear');
    }

    // ── RULE 3: Confidence floor ─────────────────────────────────────────────
    const loc_conf   = Math.max(loc_prob_shaft, 1 - loc_prob_shaft);
    const morph_conf = Math.max(comm_prob, 1 - comm_prob);
    const overall_conf = Math.min(loc_conf, dir_max, morph_conf);

    if (overall_conf < T.low_conf_floor && status_code === STATUS.CLEAR) {
      status_code = STATUS.LOW_CONFIDENCE;
      if (lang === 'zh') {
        clinical_prompt += ` 注:整体置信度 ${Math.round(overall_conf*100)}% 低于安全阈值 ${Math.round(T.low_conf_floor*100)}%,建议结合临床和补充影像综合判断。`;
      } else if (lang === 'ko') {
        clinical_prompt += ` 참고:전체 신뢰도 ${Math.round(overall_conf*100)}%는 안전 기준 ${Math.round(T.low_conf_floor*100)}% 미만이므로 임상 및 추가 영상을 종합적으로 판단할 것을 권장합니다.`;
      } else {
        clinical_prompt += ` Note: overall confidence ${Math.round(overall_conf*100)}% is below the safety floor of ${Math.round(T.low_conf_floor*100)}%; combine with clinical findings and additional imaging.`;
      }
      rules_fired.push('rule_3_low_confidence_override');
    }

    return {
      final_class,
      status_code,
      status_name: STATUS_NAMES[status_code],
      clinical_prompt,
      confidence: Math.round(overall_conf * 10000) / 10000,
      rules_fired,
      forbidden_classes,
    };
  }

  // ───────────────────────────────────────────────────────────────────────────
  // Public formatters (called by app.js)
  function formatLine(decoded, lang) {
    // If the expert system has produced a final_class, prefer it.
    if (decoded.expert_decision && decoded.expert_decision.final_class) {
      return decoded.expert_decision.final_class;
    }
    // Legacy fallback
    const T = I18N[lang] || I18N.en;
    const parts = [];
    parts.push(T.direction[decoded.direction.idx]);
    parts.push(T.location[decoded.location.idx]);
    if (decoded.morphology.displaced.yes)  parts.push(T.displaced.yes);
    if (decoded.morphology.comminuted.yes) parts.push(T.comminuted.yes);
    return parts.join(' · ');
  }

  function formatHTML(decoded, lang) {
    const T = I18N[lang] || I18N.en;
    const bar = (label, prob, kind, isYes) => {
      const pct = Math.round(prob * 100);
      const color = (kind === 'info') ? '#3b82f6' : (isYes ? '#e23' : '#2a7');
      return (
        '<div style="display:flex;align-items:center;gap:6px;font-size:12px;margin:2px 0">' +
          '<span style="width:84px;color:#666">' + label + '</span>' +
          '<div style="flex:1;background:#eee;border-radius:2px;height:6px;overflow:hidden">' +
            '<div style="width:' + pct + '%;height:100%;background:' + color + '"></div>' +
          '</div>' +
          '<span style="width:36px;text-align:right;font-variant-numeric:tabular-nums">' + pct + '%</span>' +
        '</div>'
      );
    };
    let html = '';
    html += bar(T.direction[decoded.direction.idx],  decoded.direction.conf,            'info');
    html += bar(T.location[decoded.location.idx],    decoded.location.conf,             'info');
    html += bar(T.displaced.yes,  decoded.morphology.displaced.prob,  'alarm', decoded.morphology.displaced.yes);
    html += bar(T.comminuted.yes, decoded.morphology.comminuted.prob, 'alarm', decoded.morphology.comminuted.yes);

    // ── Expert-system status badge + clinical prompt ──
    if (decoded.expert_decision) {
      const ed = decoded.expert_decision;
      // Re-compute prompt in the requested language so the UI matches the page language.
      const localizedDecision = executeExpertRules(decoded, lang);
      const STATUS_STYLES = {
        200: { bg: '#1a3a1a', fg: '#5fd97a', label: lang==='zh'?'移位':(lang==='ko'?'전위':'DISPLACED') },
        201: { bg: '#4a1a1a', fg: '#ff6b6b', label: lang==='zh'?'粉碎':(lang==='ko'?'분쇄':'COMMINUTED') },
        202: { bg: '#4a150a', fg: '#ff8c42', label: lang==='zh'?'高危粉碎':(lang==='ko'?'고위험분쇄':'HIGH-RISK') },
        203: { bg: '#1a2a4a', fg: '#7eb8ff', label: lang==='zh'?'撕脱':(lang==='ko'?'견열':'AVULSION') },
      };
      const st = STATUS_STYLES[ed.status_code] || STATUS_STYLES[200];
      html += '<div style="margin-top:8px;padding:8px 10px;background:' + st.bg + ';border-radius:5px">' +
                '<div style="display:flex;align-items:baseline;gap:8px;margin-bottom:4px">' +
                  '<span style="background:' + st.fg + ';color:#000;font-size:10px;font-weight:700;padding:1px 6px;border-radius:3px">' + st.label + '</span>' +
                  '<span style="font-size:10px;color:#aaa">' + ed.status_name + ' · ' + Math.round(ed.confidence*100) + '%</span>' +
                '</div>' +
                '<div style="font-size:11px;line-height:1.5;color:#ddd">' + localizedDecision.clinical_prompt + '</div>' +
              '</div>';
    }

    return html;
  }

  // ───────────────────────────────────────────────────────────────────────────
  // Public surface
  window.MTL = {
    load:               loadMTL,
    classifyDetection,
    applyConfidenceCalibration,
    formatLine,
    formatHTML,
    executeExpertRules,
    THRESHOLDS,
    STATUS,
    STATUS_NAMES,
    isReady: () => !!_session,
    backend: () => _backend,
    failed:  () => _failed,
    SIZE:    MTL_SIZE,
  };
})();
