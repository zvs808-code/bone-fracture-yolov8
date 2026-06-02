/* ============================================================================
 * mtl_inference.js -- Browser-side wrapper for MultiTaskFractureNet ONNX
 * Clean v7-rebuild (UTF-8 safe, no mojibake)
 * ============================================================================ */
(function () {
  'use strict';

  const MTL_PATH = 'https://media.githubusercontent.com/media/zvs808-code/bone-fracture-yolov8/main/frontend/models/model_multitask.onnx';
  const MTL_SIZE = 224;
  const MEAN = [0.485, 0.456, 0.406];
  const STD  = [0.229, 0.224, 0.225];

  const THRESHOLDS = {
    loc_shaft_lock:      0.15,
    comm_high:           0.65,
    comm_mid_low:        0.35,
    dir_transverse_lead: 0.50,
    disp_threshold:      0.50,
    low_conf_floor:      0.50,
    fragment_att_warn:   0.50,
  };

  const STATUS = {
    NORMAL_DISPLACED:     200,
    COMMINUTED:           201,
    HIGH_RISK_COMMINUTED: 202,
    AVULSION_DETECTED:    203,
    CLEAR:                200,
    AMBIGUOUS_WARNING:    202,
    LOW_CONFIDENCE:       203,
  };
  const STATUS_NAMES = {
    200: 'STATUS_NORMAL_DISPLACED',
    201: 'STATUS_COMMINUTED',
    202: 'STATUS_HIGH_RISK_COMMINUTED',
    203: 'STATUS_AVULSION_DETECTED',
  };

  const I18N = {
    ko: {
      location:   { 0: '구간(Shaft)', 1: '관절(Joint)' },
      direction:  { 0: '황상(Transverse)', 1: '사상(Oblique)', 2: '종상(Longitudinal)' },
      displaced:  { yes: '전위 있음', no: '전위 없음' },
      comminuted: { yes: '분쇄 있음', no: '분쇄 없음' },
      noFragment: '독립 골편 증거 약함',
      withDisp:   '전위 동반',
      comminutedClass: '분쇄성 골절',
    },
    en: {
      location:   { 0: 'Shaft', 1: 'Joint' },
      direction:  { 0: 'Transverse', 1: 'Oblique', 2: 'Longitudinal' },
      displaced:  { yes: 'Displaced', no: 'Non-displaced' },
      comminuted: { yes: 'Comminuted', no: 'Not comminuted' },
      noFragment: 'No independent fragment evidence -- low confidence in Comminuted',
      withDisp:   'with displacement',
      comminutedClass: 'Comminuted fracture',
    },
    zh: {
      location:   { 0: '骨干', 1: '关节' },
      direction:  { 0: '横形', 1: '斜形', 2: '纵形' },
      displaced:  { yes: '移位', no: '无明显移位' },
      comminuted: { yes: '粉碎', no: '非粉碎' },
      noFragment: '未见独立游离骨片 -- 粉碎判定低置信',
      withDisp:   '伴移位',
      comminutedClass: '粉碎性骨折',
    },
  };

  let _session = null;
  let _loading = false;
  let _failed  = false;
  let _backend = null;

  function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }
  function softmax(arr) {
    const m = Math.max.apply(null, arr);
    const ex = arr.map(v => Math.exp(v - m));
    const s = ex.reduce((a, b) => a + b, 0);
    return ex.map(v => v / s);
  }

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
    decoded.expert_decision = executeExpertRules(decoded);
    decoded._inference_ms = Math.round(performance.now() - t0);
    return decoded;
  }

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
      location:   { idx: locIdx, prob: locProbJoint, conf: locConf },
      direction:  { idx: dirIdx, probs: dirProbs, conf: dirConf },
      morphology: {
        displaced:  { prob: dispProb,         yes: dispProb        >= THRESHOLDS.disp_threshold },
        comminuted: { prob: commProb_gated,   prob_raw: commProb_raw,
                      yes:  commProb_gated   >= THRESHOLDS.disp_threshold },
      },
      fragment_att: fragmentAtt,
      task_location: { shaft_prob: locProbShaft, joint_prob: locProbJoint },
      task_direction: { transverse: dirProbs[0], oblique: dirProbs[1], longitudinal: dirProbs[2] },
      task_morphology: {
        displacement_prob:     dispProb,
        comminuted_prob_raw:   commProb_raw,
        comminuted_prob_gated: commProb_gated,
        fragment_attention:    fragmentAtt,
      },
    };
  }

  function executeExpertRules(decoded, lang) {
    lang = lang || 'zh';
    const T = THRESHOLDS;
    const L = I18N[lang] || I18N.en;
    const rules_fired = [];
    const forbidden_classes = [];
    const loc_prob_shaft = decoded.task_location.shaft_prob;
    const dir_probs = [decoded.task_direction.transverse, decoded.task_direction.oblique, decoded.task_direction.longitudinal];
    const dir_idx = dir_probs.indexOf(Math.max.apply(null, dir_probs));
    const dir_max = dir_probs[dir_idx];
    const dir_name = L.direction[dir_idx];
    const is_transverse_lead = (dir_idx === 0 && dir_max >= T.dir_transverse_lead);
    const is_shaft_strong = loc_prob_shaft >= (1.0 - T.loc_shaft_lock);
    const location_name = loc_prob_shaft >= 0.5 ? L.location[0] : L.location[1];
    const disp_prob = decoded.task_morphology.displacement_prob;
    const comm_raw = decoded.task_morphology.comminuted_prob_raw;
    const fragment_att = decoded.task_morphology.fragment_attention;
    const comm_prob = decoded.task_morphology.comminuted_prob_gated;
    const has_displacement = disp_prob >= T.disp_threshold;

    if (is_shaft_strong) {
      forbidden_classes.push('Avulsion');
      rules_fired.push('rule_1_shaft_lock_forbid_avulsion');
    }

    let final_class, status_code, clinical_prompt;
    const dispSuffix = has_displacement ? ' / ' + L.withDisp : '';

    if (comm_prob >= T.comm_high) {
      final_class = L.comminutedClass + ' / ' + location_name + dispSuffix;
      status_code = STATUS.COMMINUTED;
      clinical_prompt = 'AI: Comminuted fracture. Raw=' + Math.round(comm_raw*100) + '%, fragment_att=' + Math.round(fragment_att*100) + '%. Radiologist should confirm fragment displacement and evaluate ORIF.';
      rules_fired.push('rule_2a_comminuted_strong');
    } else if (comm_prob >= T.comm_mid_low) {
      final_class = dir_name + ' / ' + location_name + dispSuffix;
      status_code = STATUS.AMBIGUOUS_WARNING;
      clinical_prompt = 'WARN boundary: model saw rough break-line (raw ' + Math.round(comm_raw*100) + '%) but EdgeGuidedAttention found no clear independent fragment (att ' + Math.round(fragment_att*100) + '%). Radiologist should review original DICOM for 1-3mm cortical fragments.';
      rules_fired.push('rule_2b_ambiguous_warning');
    } else if (is_transverse_lead) {
      final_class = L.direction[0] + ' / ' + location_name + dispSuffix;
      status_code = STATUS.CLEAR;
      clinical_prompt = 'AI: Typical transverse fracture' + (has_displacement ? ' (with displacement)' : '') + '. No comminuted evidence (comm ' + Math.round(comm_raw*100) + '%).';
      rules_fired.push('rule_2c_collapse_to_clean_transverse');
    } else {
      final_class = dir_name + ' / ' + location_name + dispSuffix;
      status_code = STATUS.CLEAR;
      clinical_prompt = 'AI: ' + final_class + '. Morphology clear, no comminuted evidence.';
      rules_fired.push('rule_default_clear');
    }

    const loc_conf = Math.max(loc_prob_shaft, 1 - loc_prob_shaft);
    const morph_conf = Math.max(comm_prob, 1 - comm_prob);
    const overall_conf = Math.min(loc_conf, dir_max, morph_conf);
    if (overall_conf < T.low_conf_floor && status_code === STATUS.CLEAR) {
      status_code = STATUS.LOW_CONFIDENCE;
      clinical_prompt += ' NOTE: overall confidence ' + Math.round(overall_conf*100) + '% below safety floor.';
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

  function formatLine(decoded, lang) {
    if (decoded.expert_decision && decoded.expert_decision.final_class) {
      return decoded.expert_decision.final_class;
    }
    const T = I18N[lang] || I18N.en;
    const parts = [];
    parts.push(T.direction[decoded.direction.idx]);
    parts.push(T.location[decoded.location.idx]);
    if (decoded.morphology.displaced.yes)  parts.push(T.displaced.yes);
    if (decoded.morphology.comminuted.yes) parts.push(T.comminuted.yes);
    return parts.join(' / ');
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
    if (decoded.expert_decision) {
      const ed = decoded.expert_decision;
      const STATUS_STYLES = {
        200: { bg:'#1a3a1a', fg:'#5fd97a', label:'CLEAR' },
        201: { bg:'#4a1a1a', fg:'#ff6b6b', label:'COMMINUTED' },
        202: { bg:'#3a2a0a', fg:'#ffb347', label:'WARN' },
        203: { bg:'#1f2a3a', fg:'#9bb0ff', label:'LOW_CONF' },
      };
      const st = STATUS_STYLES[ed.status_code] || STATUS_STYLES[200];
      html += '<div style="margin-top:8px;padding:8px 10px;background:' + st.bg + ';border-radius:5px">' +
                '<div style="display:flex;align-items:baseline;gap:8px;margin-bottom:4px">' +
                  '<span style="background:' + st.fg + ';color:#000;font-size:10px;font-weight:700;padding:1px 6px;border-radius:3px">' + st.label + '</span>' +
                  '<span style="font-size:10px;color:#aaa">' + ed.status_name + ' / ' + Math.round(ed.confidence*100) + '%</span>' +
                '</div>' +
                '<div style="font-size:11px;line-height:1.5;color:#ddd">' + ed.clinical_prompt + '</div>' +
              '</div>';
    }
    return html;
  }

  window.MTL = {
    load: loadMTL,
    classifyDetection,
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
