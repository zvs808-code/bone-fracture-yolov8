/* ============================================================================
 * mtl_inference.js ??Browser-side wrapper for MultiTaskFractureNet ONNX
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
 *   location       Float32 [1]            BCE logit  (sigmoid > 0.5 ??Joint else Shaft)
 *   direction      Float32 [1, 3]         3-way logits (0=Transverse,1=Oblique,2=Longitudinal)
 *   morphology     Float32 [1, 2]         multi-label logits ([Displaced, Comminuted])
 *   fragment_att   Float32 [1, 1]         in [0,1] ??fragment evidence gate
 *   edge_map       Float32 [1, 1, H, W]   Sobel edge visualization
 *
 * This file is loaded via classic <script> tag in index.html AFTER ort.min.js,
 * so it just declares everything on window.MTL ??no module system needed.
 * ============================================================================ */
(function () {
  'use strict';

  const MTL_PATH   = './models/model_multitask.onnx';
  const MTL_SIZE   = 224;
  const MEAN       = [0.485, 0.456, 0.406];
  const STD        = [0.229, 0.224, 0.225];

  // ?? Clinical thresholds (MUST match Python inference_engine.py defaults) ???
  const THRESHOLDS = {
    loc_shaft_lock:      0.15,  // Rule 1: joint_prob below this = "definitely shaft"
    comm_high:           0.65,  // Rule 2a: strong comminuted floor
    comm_mid_low:        0.35,  // Rule 2b: ambiguous warning band
    dir_transverse_lead: 0.50,  // Rule 2c: transverse must lead by this
    disp_threshold:      0.50,
    low_conf_floor:      0.50,
    fragment_att_warn:   0.50,
  };

  // ?? Status codes (mirrors Python Status) ????????????????????????????????????
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

  // ?? i18n strings ????????????????????????????????????????????????????????????
  const I18N = {
    ko: {
      location:   { 0: '怨④컙(Shaft)',     1: '愿?덈?(Joint)' },
      direction:  { 0: '?≪긽(Transverse)', 1: '?ъ긽(Oblique)', 2: '醫낆긽(Longitudinal)' },
      displaced:  { yes: '?꾩쐞 ?덉쓬',       no: '?꾩쐞 ?놁쓬' },
      comminuted: { yes: '遺꾩뇙 ?덉쓬',       no: '遺꾩뇙 ?놁쓬' },
      noFragment: '?낅┰ 怨⑦렪 利앷굅 ?쏀븿 ??遺꾩뇙 吏꾨떒 ?좊ː????쓬',
      withDisp:   '?꾩쐞 ?숇컲',
      comminutedClass: '遺꾩뇙??怨⑥젅',
      ruleNames: { shaftLock: '怨④컙 ?좉툑', strongComm: '媛뺥븳 遺꾩뇙 利앷굅', ambigComm: '寃쎄퀎 寃쎄퀬', collapseTrans: '?≪긽 ?섎졃', defaultClear: '紐낇솗 ?먯젙', lowConf: '??좊ː ?ш??? },
      calib: {
        highRiskClass: '以묒쬆 ?ъ긽 遺꾩뇙??怨⑥젅(以묒쬆 ?꾩쐞 ?숇컲)',
        avulsionClass: '愿?덈궡 ?≫삎 寃ъ뿴 怨⑥젅(?꾩쐞 ?숇컲)',
        normalDispClass: '?ъ긽 怨⑥젅 쨌 怨④컙(?꾩쐞 ?숇컲)',
        highRiskPrompt: 'AI ?먯젙:**以묒쬆 ?ъ긽 遺꾩뇙??怨⑥젅**. ?섎퉬??怨⑦렪(butterfly fragment) 媛?μ꽦???믪뒿?덈떎. **臾쇰━移섎즺 湲덇린** ???뺣났쨌?닿퀬?????곴레???섏닠??怨좎젙 諛??좉꼍?덇? ?곹깭 ?됯?瑜??곗꽑?섏떗?쒖삤.',
        avulsionPrompt: 'AI ?먯젙:**愿?덈궡 ?≫삎 寃ъ뿴 怨⑥젅**. ?섏쨪/?몃? 寃ъ씤???섑븳 怨⑦렪 ?댄깉 ?⑦꽩?낅땲?? 寃ъ뿴遺 怨좎젙 諛??몃? ?먯긽 ?됯?媛 ?꾩슂?섎ŉ, 遺꾩뇙 ?뺣쪧? 寃쎄퀎 嫄곗튌湲??몄씠利덈줈 ?듭젣?섏뿀?듬땲??',
        normalDispPrompt: 'AI ?먯젙:**?쇰컲 怨④컙 ?꾩쐞 怨⑥젅**. 遺꾩뇙 ?좏샇???듭젣?섏뿀?쇰ŉ, 二?蹂묐?? ?≫삎/?꾩쐞 ?⑦꽩?낅땲?? ?앷퀬/?닿퀬????**?쇱긽???ы솢** ?꾨줈?좎퐳???곕? ???덉뒿?덈떎.',
      },
    },
    en: {
      location:   { 0: 'Shaft',         1: 'Joint' },
      direction:  { 0: 'Transverse',    1: 'Oblique', 2: 'Longitudinal' },
      displaced:  { yes: 'Displaced',   no: 'Non-displaced' },
      comminuted: { yes: 'Comminuted',  no: 'Not comminuted' },
      noFragment: 'No independent fragment evidence ??low confidence in Comminuted',
      withDisp:   'with displacement',
      comminutedClass: 'Comminuted fracture',
      ruleNames: { shaftLock: 'Shaft lock', strongComm: 'Strong comminuted', ambigComm: 'Ambiguous warning', collapseTrans: 'Collapse to transverse', defaultClear: 'Default clear', lowConf: 'Low-confidence flag' },
      calib: {
        highRiskClass: 'Severe oblique comminuted fracture (severe displacement)',
        avulsionClass: 'Intra-articular transverse avulsion (with displacement)',
        normalDispClass: 'Shaft fracture with displacement (non-comminuted)',
        highRiskPrompt: 'AI verdict: **Severe oblique comminuted fracture**. High likelihood of butterfly fragment. **Strict contraindication for physical therapy** ??prioritize ORIF and neurovascular assessment before rehab.',
        avulsionPrompt: 'AI verdict: **Intra-articular transverse avulsion fracture**. Pattern consistent with tendon/ligament traction avulsion. Secure fixation and ligament evaluation required; comminuted probability suppressed as border roughness noise.',
        normalDispPrompt: 'AI verdict: **Standard displaced diaphyseal fracture**. Comminuted signal suppressed; primary pattern is displacement/transverse. Follow routine post-immobilization rehabilitation protocol.',
      },
    },
    zh: {
      location:   { 0: '謠ⓨ묾',           1: '?녘뒄' },
      direction:  { 0: '與ゅ숱',           1: '?쒎숱', 2: '瀛드숱' },
      displaced:  { yes: '燁삡퐤',         no: '?졿삇?양㎉鵝? },
      comminuted: { yes: '暎됬쥙',         no: '?욅쾳閻? },
      noFragment: '?よ쭅?х쳦歷며┿謠①뎴 ??暎됬쥙?ㅵ츣鵝롧쉰岳?,
      withDisp:   '鴉당㎉鵝?,
      comminutedClass: '暎됬쥙?㏝え??,
      ruleNames: { shaftLock: '謠ⓨ묾?곩츣', strongComm: '凉븀쾳閻롨칮??, ambigComm: '渦밭븣鈺?몜', collapseTrans: '?뜻븲訝뷸Ø壤?, defaultClear: '?롧‘?ㅵ츣', lowConf: '鵝롧쉰岳▼쨳若? },
      calib: {
        highRiskClass: '訝ι뇥?쒎숱暎됬쥙?㏝え?섓펷鴉닻뇥佯?㎉鵝랃펹',
        avulsionClass: '?녘뒄?끾Ø壤€뮆?깁え?섓펷鴉당㎉鵝랃펹',
        normalDispClass: '謠ⓨ묾燁삡퐤謠ⓩ뒛竊덆씆暎됬쥙竊?,
        highRiskPrompt: 'AI ?ㅵ츣:**訝ι뇥?쒎숱暎됬쥙?㏝え??*?귡쳵佯??묋씢壤?え??butterfly fragment)??*?⑴릤亦사뼏訝ζ졏獵곩퓣** ??窈삡폍?덅칱鴉겼늾凉鸚띴퐤?끻쎓若?ORIF)?딁쪥瀯뤺?嶸←듁??獵곫??③え?섉쑋葉녑츣?띹퓵烏뚦볜鸚띹?瀯껁?,
        avulsionPrompt: 'AI ?ㅵ츣:**?녘뒄?끾Ø壤€뮆?깁え??*?귞Е?덅굦???㎩를?득땳?뺠꽦與▼폀,?瑥꾡섟?뺠꽦?쀥쎓若싧룋?㎩를?잋샴;暎됬쥙礖귞럤藥꿩뙃渦밭븣驪쏁퀥?ゅ０?묈댍??,
        normalDispPrompt: 'AI ?ㅵ츣:**?뉐뇛謠ⓨ묾燁삡퐤謠ⓩ뒛**?귛럴?묈댍暎됬쥙?뉐쇗뵸?쎾솵鶯?訝삣푳穩▽맏燁삡퐤/與ゅ숱謠ⓩ뒛?귞윹?뤸닑?끻쎓若싧릮??뙃**躍멱쭊?뜹뒯?롥볜鸚?*役곭쮮鸚꾤릤??,
      },
    },
  };

  // ?? Internal state ??????????????????????????????????????????????????????????
  let _session = null;
  let _loading = false;
  let _failed  = false;
  let _backend = null;

  // ?? Math helpers ????????????????????????????????????????????????????????????
  function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }
  function softmax(arr) {
    const m = Math.max.apply(null, arr);
    const ex = arr.map(v => Math.exp(v - m));
    const s = ex.reduce((a, b) => a + b, 0);
    return ex.map(v => v / s);
  }

  // ???????????????????????????????????????????????????????????????????????????
  // Load ONNX with WebGPU ??WebGL ??WASM fallback
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

  // ???????????????????????????????????????????????????????????????????????????
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

  // ???????????????????????????????????????????????????????????????????????????
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

  // ???????????????????????????????????????????????????????????????????????????
  // Decode raw ONNX outputs ??structured result (mirrors Python decode_outputs).
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
      // ?? Backward-compat shape (used by app.js renderClassification) ??
      location:   { idx: locIdx, prob: locProbJoint, conf: locConf },
      direction:  { idx: dirIdx, probs: dirProbs, conf: dirConf },
      morphology: {
        displaced:  { prob: dispProb,         yes: dispProb        >= THRESHOLDS.disp_threshold },
        comminuted: { prob: commProb_gated,   prob_raw: commProb_raw,
                      yes:  commProb_gated   >= THRESHOLDS.disp_threshold },
      },
      fragment_att: fragmentAtt,

      // ?? New ECharts-ready structured shape (mirrors Python API) ??
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

  // ?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧??  // CONFIDENCE CALIBRATION & FEATURE LINKING MATRIX
  // Realigns raw morphology logits with radiological logic (mean-convergence fix).
  // Runs immediately after decode(), before executeExpertRules().
  // ?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧??  function applyConfidenceCalibration(decoded) {
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

    // 1) Long-bone oblique displaced high-energy ??elevate true comminuted
    if (dir_oblique > 0.85 && displacement_prob > 0.95 && location_shaft > 0.90) {
      if (comminuted_raw >= 0.45) {
        comminuted_cal = 0.82;
        preset = 'high_risk_comminuted';
      }
    }
    // 2) Intra-articular avulsion ??suppress comminuted border noise
    else if (location_joint > 0.85 && dir_transverse > 0.80) {
      comminuted_cal = 0.24;
      preset = 'avulsion';
    }
    // 3) Standard non-comminuted displaced shaft ??suppress ambiguous comm band
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

  // ?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧??  // EXPERT SYSTEM ??clinical post-processing (mirrors Python execute_expert_rules)
  // ?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧??  function executeExpertRules(decoded, lang) {
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
    // Calibrated prob (post matrix) when present; else gated = raw 횞 fragment_att.
    const comm_prob      = decoded.task_morphology.comminuted_prob_calibrated != null
      ? decoded.task_morphology.comminuted_prob_calibrated
      : decoded.task_morphology.comminuted_prob_gated;
    const has_displacement = disp_prob >= T.disp_threshold;

    // ?? RULE 1: Shaft lock ???????????????????????????????????????????????????
    if (is_shaft_strong) {
      forbidden_classes.push('Avulsion');
      forbidden_classes.push(lang === 'zh' ? '?뺠꽦謠ⓩ뒛' : (lang === 'ko' ? '寃ъ뿴 怨⑥젅' : 'Avulsion fracture'));
      rules_fired.push('rule_1_shaft_lock_forbid_avulsion');
    }

    // ?? RULE 2: Dynamic Comminuted bands ?????????????????????????????????????
    let final_class, status_code, clinical_prompt;
    const dispSuffix = has_displacement ? ' 쨌 ' + L.withDisp : '';

    if (comm_prob >= T.comm_high) {
      // 2a) Strong comminuted
      final_class = L.comminutedClass + ' 쨌 ' + location_name + dispSuffix;
      status_code = STATUS.COMMINUTED;
      if (lang === 'zh') {
        clinical_prompt = `AI ?ㅵ츣:**暎됬쥙?㏝え??*(鸚싩쥙??raw 礖귞럤 ${Math.round(comm_raw*100)}%,閻롧뎴瑥곫뜮 ${Math.round(fragment_att*100)}%)?귛뻠溫?쉽?뤹쭛?삣툑簾??謠①뎴燁삡퐤亮끻벧,瑥꾡섟??맔?誤곩늾凉鸚띴퐤?끻쎓若?ORIF)?귛쫩?덂뭉曄욅퍘烏嶸→뜜鴉ㅴ퐪孃?鴉섇뀍榮㎪δ폏瑥듽?;
      } else if (lang === 'ko') {
        clinical_prompt = `AI ?먯젙:**遺꾩뇙??怨⑥젅**(?ㅼ쨷 怨⑦렪,raw ${Math.round(comm_raw*100)}%,怨⑦렪 利앷굅 ${Math.round(fragment_att*100)}%)?귥쁺?곸쓽???꾨Ц?섍? 怨⑦렪 ?꾩쐞 ?뺣룄瑜??뺤씤?섍퀬 ?덇컻?뺣났?닿퀬?뺤닠(ORIF) ?꾩슂?깆쓣 ?됯???寃껋쓣 沅뚯옣?⑸땲??`;
      } else {
        clinical_prompt = `AI verdict: **Comminuted fracture** (multiple fragments, raw probability ${Math.round(comm_raw*100)}%, fragment evidence ${Math.round(fragment_att*100)}%). Radiologist should confirm fragment displacement and assess need for ORIF. Emergency consult if neurovascular signs.`;
      }
      rules_fired.push('rule_2a_comminuted_strong');

    } else if (comm_prob >= T.comm_mid_low) {
      // 2b) Ambiguous range ??strong warning
      final_class = dir_name + (lang === 'zh' ? '謠ⓩ뒛' : (lang === 'ko' ? ' 怨⑥젅' : ' fracture')) + ' 쨌 ' + location_name + dispSuffix;
      status_code = STATUS.AMBIGUOUS_WARNING;
      if (lang === 'zh') {
        clinical_prompt = `??**?븃꺗渦밭븣鈺?몜**:與▼엹汝役뗥댆??룭壤€곩킂略?Comminuted raw ${Math.round(comm_raw*100)}%),鵝?EdgeGuidedAttention ?ゅ룕?경삇簾?떖塋뗦만獵삯え??fragment_att ${Math.round(fragment_att*100)}%,鵝롣틢 ${Math.round(T.fragment_att_warn*100)}% ?덂??귛뻠溫?붂弱꾤쭛?삣툑?ⓨ렅冶?DICOM 訝듾퍝瀯녶쨳?멩?鸚꾣삸??춼??1?? mm ?꾤퍏孃?슢兀①쥙??餓ζ럲?ㅸ슥佯?쾳閻롦㏝え??occult comminution)?귞퍜?덁릿佯딁뾽??遙섋꺗?뤸뜜鴉?)訝롨슈鵝띸뎴瀯쇔릦?ㅶ뼪??;
      } else if (lang === 'ko') {
        clinical_prompt = `??**吏?ν삎 寃쎄퀎 寃쎄퀬**:紐⑤뜽? 嫄곗튇 怨⑥젅??Comminuted raw ${Math.round(comm_raw*100)}%)??媛먯??덉쑝??EdgeGuidedAttention? 紐낇솗???낅┰ 怨⑦렪??諛쒓껄?섏? 紐삵뻽?듬땲??fragment_att ${Math.round(fragment_att*100)}%)?귥썝蹂?DICOM?먯꽌 1?? mm??誘몄꽭 ?쇱쭏 怨⑦렪??硫대????ш??좏븯??寃쎌쬆 遺꾩뇙??怨⑥젅??諛곗젣??寃껋쓣 沅뚯옣?⑸땲??`;
      } else {
        clinical_prompt = `??**Smart boundary warning**: The model detected a jagged break-line texture (Comminuted raw ${Math.round(comm_raw*100)}%) but EdgeGuidedAttention found no clear independent fragment (fragment_att ${Math.round(fragment_att*100)}%, below the ${Math.round(T.fragment_att_warn*100)}% threshold). Radiologist should carefully review the original DICOM for any 1?? mm cortical fragments to rule out occult comminution. Correlate with clinical history (high-energy trauma?) and axial views.`;
      }
      rules_fired.push('rule_2b_ambiguous_warning');

    } else if (is_transverse_lead) {
      // 2c) Low comminuted + transverse leading ??collapse
      final_class = L.direction[0] + (lang === 'zh' ? '謠ⓩ뒛' : (lang === 'ko' ? ' 怨⑥젅' : ' fracture')) + ' 쨌 ' + location_name + dispSuffix;
      status_code = STATUS.CLEAR;
      if (lang === 'zh') {
        clinical_prompt = `AI ?ㅵ츣:**?멨엹與ゅ숱謠ⓩ뒛**${has_displacement?'(鴉당㎉鵝?':''}?귝쑋鰲곭쾳閻롦닑歷며┿謠①뎴瑥곫뜮(comm_prob ${Math.round(comm_raw*100)}% < 35%)?귛룾?됪Ø壤?え?섉젃?녶쨳鵝띷탛葉뗥쨪?녴?;
      } else if (lang === 'ko') {
        clinical_prompt = `AI ?먯젙:**?꾪삎???≪긽 怨⑥젅**${has_displacement?'(?꾩쐞 ?숇컲)':''}?귣텇?꾨굹 怨⑦렪 利앷굅 ?놁쓬(comm_prob ${Math.round(comm_raw*100)}% < 35%)?귦몴以 ?≪긽 怨⑥젅 蹂듭쐞 ?덉감瑜??곕? ???덉뒿?덈떎.`;
      } else {
        clinical_prompt = `AI verdict: **Typical transverse fracture**${has_displacement?' with displacement':''}. No comminuted or fragment evidence (comm_prob ${Math.round(comm_raw*100)}% < 35%). Can follow standard transverse-fracture reduction protocol.`;
      }
      rules_fired.push('rule_2c_collapse_to_clean_transverse');

    } else {
      // Default clean call
      final_class = dir_name + (lang === 'zh' ? '謠ⓩ뒛' : (lang === 'ko' ? ' 怨⑥젅' : ' fracture')) + ' 쨌 ' + location_name + dispSuffix;
      status_code = STATUS.CLEAR;
      if (lang === 'zh') {
        clinical_prompt = `AI ?ㅵ츣:${final_class}?귛숱?곩??밧푳?롧‘,?좂쾳閻롨칮???;
      } else if (lang === 'ko') {
        clinical_prompt = `AI ?먯젙:${final_class}?귦삎?쒗븰???뱀쭠??紐낇솗?섎ŉ 遺꾩뇙 利앷굅媛 ?놁뒿?덈떎.`;
      } else {
        clinical_prompt = `AI verdict: ${final_class}. Morphological features are clear, no comminution evidence.`;
      }
      rules_fired.push('rule_default_clear');
    }

    // ?? RULE 3: Confidence floor ?????????????????????????????????????????????
    const loc_conf   = Math.max(loc_prob_shaft, 1 - loc_prob_shaft);
    const morph_conf = Math.max(comm_prob, 1 - comm_prob);
    const overall_conf = Math.min(loc_conf, dir_max, morph_conf);

    if (overall_conf < T.low_conf_floor && status_code === STATUS.CLEAR) {
      status_code = STATUS.LOW_CONFIDENCE;
      if (lang === 'zh') {
        clinical_prompt += ` 力??답퐪營?에佯?${Math.round(overall_conf*100)}% 鵝롣틢若됧뀲?덂?${Math.round(T.low_conf_floor*100)}%,兩븃?瀯볟릦訝닷틞?뚩‥?끻쉽?뤹뻤?덂닩???;
      } else if (lang === 'ko') {
        clinical_prompt += ` 李멸퀬:?꾩껜 ?좊ː??${Math.round(overall_conf*100)}%???덉쟾 湲곗? ${Math.round(T.low_conf_floor*100)}% 誘몃쭔?대?濡??꾩긽 諛?異붽? ?곸긽??醫낇빀?곸쑝濡??먮떒??寃껋쓣 沅뚯옣?⑸땲??`;
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

  // ???????????????????????????????????????????????????????????????????????????
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
    return parts.join(' 쨌 ');
  }

  function formatHTML(decoded, lang) {
    const T = I18N[lang] || I18N.en;
    const bar = (label, prob, kind, isYes) => {
      const pct = Math.round(prob * 100);
      const fillClass = kind === 'info' ? 'info' : (isYes ? 'alarm-yes' : 'alarm-no');
      return (
        '<div class="mtl-bar">' +
          '<span class="mtl-bar-label">' + label + '</span>' +
          '<div class="mtl-bar-track">' +
            '<div class="mtl-bar-fill ' + fillClass + '" style="width:' + pct + '%"></div>' +
          '</div>' +
          '<span class="mtl-bar-pct">' + pct + '%</span>' +
        '</div>'
      );
    };
    let html = '';
    html += bar(T.direction[decoded.direction.idx], decoded.direction.conf, 'info');
    html += bar(T.location[decoded.location.idx], decoded.location.conf, 'info');
    html += bar(T.displaced.yes, decoded.morphology.displaced.prob, 'alarm', decoded.morphology.displaced.yes);
    html += bar(T.comminuted.yes, decoded.morphology.comminuted.prob, 'alarm', decoded.morphology.comminuted.yes);

    if (decoded.expert_decision) {
      const ed = decoded.expert_decision;
      const localizedDecision = executeExpertRules(decoded, lang);
      const STATUS_STYLES = {
        200: { cls: 'mtl-verdict--disp', label: lang === 'zh' ? '燁삡퐤' : (lang === 'ko' ? '?꾩쐞' : 'DISPLACED') },
        201: { cls: 'mtl-verdict--comm', label: lang === 'zh' ? '暎됬쥙' : (lang === 'ko' ? '遺꾩뇙' : 'COMMINUTED') },
        202: { cls: 'mtl-verdict--risk', label: lang === 'zh' ? '遙섇뜳暎됬쥙' : (lang === 'ko' ? '怨좎쐞?섎텇?? : 'HIGH-RISK') },
        203: { cls: 'mtl-verdict--avulsion', label: lang === 'zh' ? '?뺠꽦' : (lang === 'ko' ? '寃ъ뿴' : 'AVULSION') },
      };
      const st = STATUS_STYLES[ed.status_code] || STATUS_STYLES[200];
      const prompt = (localizedDecision.clinical_prompt || '').replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
      html +=
        '<div class="mtl-verdict ' + st.cls + '">' +
          '<div class="mtl-verdict-hd">' +
            '<span class="mtl-verdict-badge">' + st.label + '</span>' +
            '<span class="mtl-verdict-meta">' + ed.status_name + ' 쨌 ' + Math.round(ed.confidence * 100) + '%</span>' +
          '</div>' +
          '<div class="mtl-verdict-body">' + prompt + '</div>' +
        '</div>';
    }

    return html;
  }

  // ???????????????????????????????????????????????????????????????????????????
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
