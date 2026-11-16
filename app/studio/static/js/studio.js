/* ACF Scan Studio, real-data edition. Ported from app/studio/reference/acf_studio.html: same volume object
   V = {nx, ny, nz, sx, sy, sz, ct, pet, lab, hasPet, hasLab, flip, real, name}, same drawing code.
   What differs from the reference, and why:
   - a real case comes from the local backend (/api/cases/<id>/volume), not from a file picker;
   - the demo evidence rule (evid) is used on the synthetic phantom ONLY; a real case never shows a demo number;
   - PET of a real case is the released value divided by the case's own scale (what the network is given), so
     every PET display constant of the reference is multiplied by V.petK (1 for the phantom, 0.1 for a real case:
     the phantom's "SUV 10" plays the role of 1.0 x case scale);
   - the 3D block draws anterior on the near edge, so the head faces the viewer and left and right are not mirrored;
   - Fusion can show three planes at once as well as the reference's CT / PET / fused row. */
(function () {
  'use strict';
  var $ = function (id) { return document.getElementById(id); };
  // what the page is showing and any script error, readable from the DOM (used by the headless checks)
  function noteError(m) { document.body.setAttribute('data-errors', (document.body.getAttribute('data-errors') || '') + '[' + m + ']'); }
  window.addEventListener('error', function (e) { noteError(e.message || 'error'); });
  window.addEventListener('unhandledrejection', function (e) { noteError('rejection: ' + (e.reason && e.reason.message || e.reason)); });
  function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }
  function smooth(t) { t = clamp(t, 0, 1); return t * t * (3 - 2 * t); }
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); }
  function hash(i, j, k) {
    var h = (i * 374761393 + j * 668265263 + k * 2147483647) | 0;
    h = (h ^ (h >>> 13)) * 1274126177 | 0; h = h ^ (h >>> 16);
    return ((h & 0xffff) / 32768) - 1;
  }
  function vnoise(x, y, z) {
    var xi = Math.floor(x), yi = Math.floor(y), zi = Math.floor(z), fx = x - xi, fy = y - yi, fz = z - zi;
    fx = fx * fx * (3 - 2 * fx); fy = fy * fy * (3 - 2 * fy); fz = fz * fz * (3 - 2 * fz);
    function L(a, b, c) { return hash(xi + a, yi + b, zi + c); }
    var x00 = L(0, 0, 0) + (L(1, 0, 0) - L(0, 0, 0)) * fx, x10 = L(0, 1, 0) + (L(1, 1, 0) - L(0, 1, 0)) * fx;
    var x01 = L(0, 0, 1) + (L(1, 0, 1) - L(0, 0, 1)) * fx, x11 = L(0, 1, 1) + (L(1, 1, 1) - L(0, 1, 1)) * fx;
    var y0 = x00 + (x10 - x00) * fy, y1 = x01 + (x11 - x01) * fy;
    return y0 + (y1 - y0) * fz;
  }

  // ---------- evidence rule (demo, synthetic phantom only): evidence from PET uptake ----------
  function evid(s) {
    var tt = 4.5;
    return [12 * smooth((tt - s) / 3), 12 * smooth((s - tt) / 3)];   // [e0 background, e1 tumour]
  }

  // ---------- synthetic phantom, sampled at any spacing ----------
  var WATERMARK = 'Research use only. HECKTOR data. Not for publication.';
  var NAMES = { 0: 'background', 1: 'primary tumour', 2: 'lymph node' };          // phantom only
  var REAL_NAMES = { 0: 'background (0)', 1: 'label 1', 2: 'label 2' };           // meaning not verified
  var BLOBS = [
    { c: [74, 106, 66], r: [11, 10, 12], peak: 11, lab: 1 },
    { c: [130, 90, 58], r: [8, 8, 9], peak: 8, lab: 2 },
    { c: [62, 90, 50], r: [6, 6, 7], peak: 6, lab: 2 }
  ];
  function ell(x, y, z, c, r) { var a = (x - c[0]) / r[0], b = (y - c[1]) / r[1], d = (z - c[2]) / r[2]; return a * a + b * b + d * d; }
  function gen(sx, sy, sz) {
    var EXT = [192, 192, 144];
    var nx = Math.round(EXT[0] / sx), ny = Math.round(EXT[1] / sy), nz = Math.round(EXT[2] / sz), n = nx * ny * nz;
    var ct = new Int16Array(n), pet = new Float32Array(n), lab = new Uint8Array(n), idx = 0;
    for (var k = 0; k < nz; k++) {
      var z = (k + 0.5) * sz, t = smooth((z - 50) / 22), dome = Math.sqrt(Math.max(0.02, 1 - Math.pow(clamp((z - 96) / 48, 0, 1), 2.2))), rx = (40 + 30 * t) * dome, ry = (46 + 40 * t) * dome;
      for (var j = 0; j < ny; j++) {
        var y = (j + 0.5) * sy;
        for (var i = 0; i < nx; i++, idx++) {
          var x = (i + 0.5) * sx, dx = (x - 96) / rx, dy = (y - 96) / ry, q = dx * dx + dy * dy, hu = -1000, suv = 0.05, L = 0;
          if (q < 1) {
            hu = 42 + 10 * vnoise(x / 7, y / 7, z / 7); suv = 1.1 + 0.25 * vnoise(x / 9, y / 9, z / 9);
            if (z < 78 && z > 14) {                                                         // sternocleidomastoid
              if (ell(x, y, z, [60, 92, 46], [9, 10, 32]) < 1 || ell(x, y, z, [132, 92, 46], [9, 10, 32]) < 1) { hu = 58 + 6 * vnoise(x / 4, y / 4, z / 4); suv = 1.7; }
            }
            if (q > 0.8 && q < 0.93 && z > 75) { hu = 620 + 120 * vnoise(x / 5, y / 5, z / 5); suv = 0.6; }   // skull
            else if (q < 0.8 && z > 82) {                                                   // brain
              hu = 32 + 9 * vnoise(x / 4, y / 4, z / 4); suv = 3.3 + 0.5 * vnoise(x / 6, y / 6, z / 6);
              if (ell(x, y, z, [96, 98, 104], [7, 17, 9]) < 1) { hu = 8; suv = 0.4; }     // ventricles
            } else if (q > 0.72 && q < 0.9 && z > 60 && z < 78 && y > 96) { hu = 580; suv = 0.6; }  // jaw
            var sdx = x - 96, sdy = y - (96 - 0.5 * ry);
            if (sdx * sdx + sdy * sdy < 81 && z < 80) { hu = 470 + 60 * vnoise(x / 4, y / 4, z / 3); suv = 1.4; }  // spine
            var adx = (x - 96) / 8, ady = (y - 98) / 10;
            if (adx * adx + ady * ady < 1 && z > 40 && z < 86) { hu = -950; suv = 0.2; }     // airway
            if (z > 62 && z < 80 && (ell(x, y, z, [80, 118, 70], [10, 8, 9]) < 1 || ell(x, y, z, [112, 118, 70], [10, 8, 9]) < 1)) { hu = -980; suv = 0.1; } // sinuses
            for (var b = 0; b < BLOBS.length; b++) {
              var B = BLOBS[b], d = ell(x, y, z, B.c, B.r);
              suv += B.peak * Math.exp(-1.8 * d);
              if (d < 1) { L = B.lab; hu = Math.max(hu, 50); }
            }
          }
          hu += 6 * hash(i, j, k);
          ct[idx] = hu; pet[idx] = Math.max(0, suv + 0.25 * hash(k, i, j)); lab[idx] = L;
        }
      }
    }
    return { nx: nx, ny: ny, nz: nz, sx: sx, sy: sy, sz: sz, ext: [nx * sx, ny * sy, nz * sz], ct: ct, pet: pet, lab: lab,
      hasPet: true, hasLab: true, flip: [false, true, true], real: false, name: 'Synthetic phantom', petK: 1 };
  }

  // ---------- a real case from the local backend ----------
  var HALF = null;
  function halfTable() {                     // IEEE half -> float, as a 65536-entry table
    if (HALF) return HALF;
    HALF = new Float32Array(65536);
    for (var h = 0; h < 65536; h++) {
      var s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
      HALF[h] = e === 0 ? s * Math.pow(2, -14) * (f / 1024) : e === 31 ? (f ? NaN : s * Infinity) : s * Math.pow(2, e - 15) * (1 + f / 1024);
    }
    return HALF;
  }
  function parseVolume(buf) {
    var u8 = new Uint8Array(buf), magic = '';
    for (var i = 0; i < 8; i++) magic += String.fromCharCode(u8[i]);
    if (magic !== 'ACFV0001') throw new Error('The server sent something that is not a display volume.');
    var hl = new DataView(buf).getUint32(8, true), H = JSON.parse(new TextDecoder('utf-8').decode(u8.subarray(12, 12 + hl)));
    var n = H.nx * H.ny * H.nz, A = H.arrays;
    if (A.ct.dtype !== 'int16' || A.pet.dtype !== 'float16' || A.label.dtype !== 'uint8') throw new Error('Unexpected array types in the display volume.');
    var ct = new Int16Array(buf, A.ct.offset, n), ph = new Uint16Array(buf, A.pet.offset, n), lab = new Uint8Array(buf, A.label.offset, n);
    var T = halfTable(), pet = new Float32Array(n);
    for (var q = 0; q < n; q++) pet[q] = T[ph[q]];
    return { nx: H.nx, ny: H.ny, nz: H.nz, sx: H.sx, sy: H.sy, sz: H.sz, ext: [H.nx * H.sx, H.ny * H.sy, H.nz * H.sz], ct: ct, pet: pet, lab: lab,
      hasPet: true, hasLab: true, flip: H.flip, real: true, name: H.patient_id, petK: 0.1, meta: H };
  }
  function labelCentre(v) {                  // start the crosshair on the label, like the reference's loader
    var sx = 0, sy = 0, sz = 0, c = 0, nxy = v.nx * v.ny, n = nxy * v.nz;
    for (var q = 0; q < n; q++) if (v.lab[q]) { sx += q % v.nx; sy += Math.floor(q / v.nx) % v.ny; sz += Math.floor(q / nxy); c++; }
    if (!c) return [v.ext[0] / 2, v.ext[1] / 2, v.ext[2] / 2];
    return [(sx / c + 0.5) * v.sx, (sy / c + 0.5) * v.sy, (sz / c + 0.5) * v.sz];
  }

  // ---------- metrics measured on the synthetic phantom (demo rule; never run on a real case) ----------
  function stats(V) {
    var n = V.nx * V.ny * V.nz, vox = V.sx * V.sy * V.sz / 1000, gs = 0, ps = 0, inter = 0, uin = 0, uout = 0, nin = 0, nout = 0;
    var cnt = new Float64Array(15), conf = new Float64Array(15), acc = new Float64Array(15), total = 0, eceSum = 0;
    if (V.real || !V.hasPet) return { ok: false };
    for (var i = 0; i < n; i++) {
      var e = evid(V.pet[i]), a1 = e[1] + 1, a0 = e[0] + 1, S = a0 + a1, p = a1 / S, u = 2 / S;
      var pr = p > 0.5, tr = V.hasLab && V.lab[i] > 0;
      if (pr) ps++;
      if (V.hasLab) {
        if (tr) { gs++; uin += u; nin++; } else { uout += u; nout++; }
        if (pr && tr) inter++;
        var c = Math.max(p, 1 - p), bi = Math.min(14, Math.floor(c * 15)), ok = (pr === tr) ? 1 : 0;
        cnt[bi]++; conf[bi] += c; acc[bi] += ok; total++;
      }
    }
    var rel = [];
    if (V.hasLab) for (var b = 0; b < 15; b++) if (cnt[b] > 0) { var ac = acc[b] / cnt[b], cf = conf[b] / cnt[b]; eceSum += (cnt[b] / total) * Math.abs(ac - cf); rel.push([cf, ac, cnt[b]]); }
    return { ok: true, hasLab: V.hasLab, dice: V.hasLab && (gs + ps) ? 2 * inter / (gs + ps) : NaN, vol: gs * vox, pvol: ps * vox,
      ece: V.hasLab ? eceSum : NaN, uin: nin ? uin / nin : NaN, uout: nout ? uout / nout : NaN, rel: rel };
  }

  // ---------- state ----------
  var V = null, ST = null;
  var st = { plane: 'axial', cur: [74, 106, 66], scene: 'volume', layout: 'planes', mode: 'fuse' };
  var CW = 420;
  var PLANES = ['axial', 'coronal', 'sagittal'], MODES = ['ct', 'pet', 'fuse'];
  var PLANE_NAME = { axial: 'Axial', coronal: 'Coronal', sagittal: 'Sagittal' }, MODE_NAME = { ct: 'CT', pet: 'PET', fuse: 'Fused' };
  // letters at the left, right, top and bottom of each plane; valid because flip[a] is true exactly where the
  // voxel index grows toward patient Right / Anterior / Superior (backend: flips_from_affine)
  var EDGE = { axial: ['R', 'L', 'A', 'P'], coronal: ['R', 'L', 'S', 'I'], sagittal: ['A', 'P', 'S', 'I'] };

  function idxOf() {
    return [clamp(Math.floor(st.cur[0] / V.sx), 0, V.nx - 1), clamp(Math.floor(st.cur[1] / V.sy), 0, V.ny - 1), clamp(Math.floor(st.cur[2] / V.sz), 0, V.nz - 1)];
  }
  function planeDims(pl) {
    if (pl === 'axial') return { w: V.nx, h: V.ny, W: V.ext[0], H: V.ext[1] };
    if (pl === 'coronal') return { w: V.nx, h: V.nz, W: V.ext[0], H: V.ext[2] };
    return { w: V.ny, h: V.nz, W: V.ext[1], H: V.ext[2] };
  }
  function pixToVox(pl, c, r, I) {
    var f = V.flip;
    if (pl === 'axial') return [f[0] ? V.nx - 1 - c : c, f[1] ? V.ny - 1 - r : r, I[2]];
    if (pl === 'coronal') return [f[0] ? V.nx - 1 - c : c, I[1], f[2] ? V.nz - 1 - r : r];
    return [I[0], f[1] ? V.ny - 1 - c : c, f[2] ? V.nz - 1 - r : r];
  }
  function mmToFrac(pl) {
    var f = V.flip, c = st.cur, fx, fy;
    if (pl === 'axial') { fx = c[0] / V.ext[0]; fy = c[1] / V.ext[1]; return [f[0] ? 1 - fx : fx, f[1] ? 1 - fy : fy]; }
    if (pl === 'coronal') { fx = c[0] / V.ext[0]; fy = c[2] / V.ext[2]; return [f[0] ? 1 - fx : fx, f[2] ? 1 - fy : fy]; }
    fx = c[1] / V.ext[1]; fy = c[2] / V.ext[2]; return [f[1] ? 1 - fx : fx, f[2] ? 1 - fy : fy];
  }
  function fracToMm(pl, fx, fy) {
    var f = V.flip, c = st.cur;
    if (pl === 'axial') { c[0] = (f[0] ? 1 - fx : fx) * V.ext[0]; c[1] = (f[1] ? 1 - fy : fy) * V.ext[1]; }
    else if (pl === 'coronal') { c[0] = (f[0] ? 1 - fx : fx) * V.ext[0]; c[2] = (f[2] ? 1 - fy : fy) * V.ext[2]; }
    else { c[1] = (f[1] ? 1 - fx : fx) * V.ext[1]; c[2] = (f[2] ? 1 - fy : fy) * V.ext[2]; }
  }
  function hot(v) { v = clamp(v, 0, 1); return [Math.min(1, v * 3), clamp(v * 3 - 1, 0, 1), clamp(v * 3 - 2, 0, 1)]; }
  function watermark(ctx, w, h) {
    ctx.save(); ctx.font = '11px ui-monospace, monospace'; ctx.textAlign = 'center';
    ctx.fillStyle = 'rgba(0,0,0,0.55)'; ctx.fillRect(0, h - 18, w, 18);
    ctx.fillStyle = 'rgba(255,180,84,0.95)'; ctx.fillText(WATERMARK, w / 2, h - 5, w - 8); ctx.restore();
  }

  // ---------- 2D plane drawing (CT, PET, fused) ----------
  function drawPlane(cv, mode, o) {
    var pl = o.plane || st.plane, I = idxOf(), d = planeDims(pl), w = d.w, h = d.h, Hc = Math.round(CW * d.H / d.W);
    var K = V.petK, img = new ImageData(w, h), vi = new Int32Array(w * h), petMax = (o.petMax || 10) * K, op = o.petOp == null ? 0.65 : o.petOp;
    var U = [154, 123, 255], hasP = V.hasPet && o.pet !== false, demo = !V.real;
    for (var r = 0; r < h; r++) for (var c = 0; c < w; c++) {
      var t = pixToVox(pl, c, r, I), id = t[0] + V.nx * (t[1] + V.ny * t[2]); vi[r * w + c] = id;
      var g = clamp((V.ct[id] + 135) / 350, 0, 1) * 255, R = g, G = g, Bc = g;
      if (mode === 'pet' && V.hasPet) { var hc = hot(V.pet[id] / petMax); R = hc[0] * 255; G = hc[1] * 255; Bc = hc[2] * 255; }
      else if (mode === 'pet') { R = G = Bc = 0; }
      else if (mode === 'fuse' && hasP) {
        var hv = hot(V.pet[id] / petMax), a = op * clamp((V.pet[id] - 1.5 * K) / (2.5 * K), 0, 1);
        R = g * (1 - a) + hv[0] * 255 * a; G = g * (1 - a) + hv[1] * 255 * a; Bc = g * (1 - a) + hv[2] * 255 * a;
      }
      if (o.unc && demo && V.hasPet) {
        var e = evid(V.pet[id]), u = 2 / (e[0] + e[1] + 2), ua = Math.max(0, u - 0.3) * 0.9;
        R = R * (1 - ua) + U[0] * ua; G = G * (1 - ua) + U[1] * ua; Bc = Bc * (1 - ua) + U[2] * ua;
      }
      var q = (r * w + c) * 4; img.data[q] = R; img.data[q + 1] = G; img.data[q + 2] = Bc; img.data[q + 3] = 255;
    }
    function edge(test, col) {
      for (var r = 0; r < h; r++) for (var c = 0; c < w; c++) {
        var i0 = r * w + c; if (!test(vi[i0])) continue;
        if ((c > 0 && !test(vi[i0 - 1])) || (c < w - 1 && !test(vi[i0 + 1])) || (r > 0 && !test(vi[i0 - w])) || (r < h - 1 && !test(vi[i0 + w]))) {
          var q = i0 * 4; img.data[q] = col[0]; img.data[q + 1] = col[1]; img.data[q + 2] = col[2];
        }
      }
    }
    if (o.pred && demo && V.hasPet) edge(function (id) { var e = evid(V.pet[id]); return e[1] > e[0]; }, [63, 216, 255]);
    if (o.gt && V.hasLab) { edge(function (id) { return V.lab[id] === 1; }, [255, 77, 77]); edge(function (id) { return V.lab[id] === 2; }, [75, 227, 154]); }
    var off = document.createElement('canvas'); off.width = w; off.height = h; off.getContext('2d').putImageData(img, 0, 0);
    cv.width = CW; cv.height = Hc;
    var ctx = cv.getContext('2d'); ctx.imageSmoothingEnabled = false; ctx.drawImage(off, 0, 0, CW, Hc);
    var f = mmToFrac(pl), px = f[0] * CW, py = f[1] * Hc;
    ctx.strokeStyle = 'rgba(63,216,255,0.8)'; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, Hc); ctx.moveTo(0, py); ctx.lineTo(CW, py); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = 'rgba(227,242,250,0.8)'; ctx.font = '11px ui-monospace, monospace';
    if (V.real) {
      var E = EDGE[pl]; ctx.fillText(E[0], 6, Hc / 2 - 4); ctx.fillText(E[1], CW - 14, Hc / 2 - 4); ctx.fillText(E[2], CW / 2 + 5, 13); ctx.fillText(E[3], CW / 2 + 5, Hc - 24);
      watermark(ctx, CW, Hc);
    } else { ctx.fillText('R', 6, Hc / 2 - 4); ctx.fillText('L', CW - 14, Hc / 2 - 4); }
  }
  function panes() {
    return [0, 1, 2].map(function (i) { return st.layout === 'planes' ? { plane: PLANES[i], mode: st.mode } : { plane: st.plane, mode: MODES[i] }; });
  }
  function sliceInfo(pl) {
    var n = pl === 'axial' ? V.nz : pl === 'coronal' ? V.ny : V.nx, sp = pl === 'axial' ? V.sz : pl === 'coronal' ? V.sy : V.sx, key = pl === 'axial' ? 2 : pl === 'coronal' ? 1 : 0;
    return { n: n, sp: sp, key: key, i: clamp(Math.floor(st.cur[key] / sp), 0, n - 1) };
  }
  function petCaption(mx) {
    return V.real ? 'PET / case scale, 0 to ' + (mx * V.petK).toFixed(2) : 'SUV 0 to ' + mx;
  }
  function renderFusion() {
    var mx = parseFloat($('pet-max').value);
    var o = { petMax: mx, petOp: parseFloat($('pet-op').value), gt: $('f-gt').checked, pred: false, unc: $('f-unc').checked && !V.real };
    panes().forEach(function (p, i) {
      drawPlane($('cv-' + i), p.mode, { plane: p.plane, petMax: o.petMax, petOp: o.petOp, gt: o.gt, pred: false, unc: o.unc });
      var s = sliceInfo(p.plane);
      $('vt-' + i).textContent = st.layout === 'planes' ? PLANE_NAME[p.plane] : MODE_NAME[p.mode];
      $('vs-' + i).textContent = st.layout === 'planes' ? 'slice ' + (s.i + 1) + ' / ' + s.n
        : p.mode === 'ct' ? 'HU, 350 / 40' : p.mode === 'pet' ? petCaption(mx) : 'PET over CT';
    });
    $('fusion-note').textContent = V.real
      ? MODE_NAME[st.mode] + ' view. PET is the released value divided by this case\'s scale ' + V.meta.pet.scale.toPrecision(5) + ' (' + V.meta.pet.note + '), clipped at ' + V.meta.pet.clip_max + '. Outlines: label 1 red, label 2 green. No uncertainty map is loaded for this case.'
      : 'Synthetic phantom. The uncertainty glow is the demo rule on PET uptake, not a model output.';
    probeInto($('probe'));
  }
  function probeInto(el) {
    var I = idxOf(), id = I[0] + V.nx * (I[1] + V.ny * I[2]), rows = [
      ['Position mm', st.cur.map(function (v) { return v.toFixed(0); }).join(', ')], ['Voxel', I.join(', ')], ['CT HU', String(V.ct[id])]];
    if (V.real) {
      rows.push(['PET / case scale', V.pet[id].toFixed(3)], ['PET as released', (V.pet[id] * V.meta.pet.scale).toPrecision(4) + ' (units not stated)'], ['Label', REAL_NAMES[V.lab[id]] || ('label ' + V.lab[id])]);
    } else if (V.hasPet) {
      var e = evid(V.pet[id]), S = e[0] + e[1] + 2;
      rows.push(['PET SUV', V.pet[id].toFixed(1)]);
      if (V.hasLab) rows.push(['Truth', NAMES[V.lab[id]] || ('label ' + V.lab[id])]);
      rows.push(['Demo p(tumour)', ((e[1] + 1) / S).toFixed(2)], ['Uncertainty u', (2 / S).toFixed(2)]);
    }
    el.innerHTML = rows.map(function (r) { return '<div class="pv"><span>' + esc(r[0]) + '</span><span>' + esc(r[1]) + '</span></div>'; }).join('');
  }
  function updateSlice() {
    var s = $('slice'), q = sliceInfo(st.plane); s.max = q.n - 1; s.value = q.i;
    $('slice-read').textContent = PLANE_NAME[st.plane] + ' ' + (q.i + 1) + ' / ' + q.n + ' (' + ((q.i + 0.5) * q.sp).toFixed(1) + ' mm)';
  }
  function syncSegs() {
    function press(box, attr, val) { Array.prototype.forEach.call(box.querySelectorAll('button'), function (x) { x.setAttribute('aria-pressed', x.getAttribute(attr) === val ? 'true' : 'false'); }); }
    press($('plane-seg'), 'data-plane', st.plane); press($('mode-seg'), 'data-mode', st.mode); press($('layout-seg'), 'data-layout', st.layout);
    $('mode-ctl').hidden = st.layout !== 'planes';
  }
  function bindPick(cv, n) {
    var down = false;
    function pick(e) {
      var rc = cv.getBoundingClientRect(), pl = panes()[n].plane;
      fracToMm(pl, clamp((e.clientX - rc.left) / rc.width, 0, 1), clamp((e.clientY - rc.top) / rc.height, 0, 1));
      if (st.layout === 'planes' && st.plane !== pl) { st.plane = pl; syncSegs(); }
      updateSlice(); renderFusion();
    }
    cv.addEventListener('pointerdown', function (e) { down = true; cv.setPointerCapture(e.pointerId); pick(e); });
    cv.addEventListener('pointermove', function (e) { if (down) pick(e); });
    cv.addEventListener('pointerup', function () { down = false; });
    cv.addEventListener('contextmenu', function (e) { if (V && V.real) e.preventDefault(); });
    cv.addEventListener('keydown', function (e) {
      var s = $('slice'), v = parseInt(s.value, 10);
      if (e.key === 'ArrowUp' || e.key === 'ArrowRight') { s.value = v + 1; s.dispatchEvent(new Event('input')); e.preventDefault(); }
      if (e.key === 'ArrowDown' || e.key === 'ArrowLeft') { s.value = v - 1; s.dispatchEvent(new Event('input')); e.preventDefault(); }
    });
  }
  function refreshScene() {
    if (st.scene === 'fusion') { syncSegs(); updateSlice(); renderFusion(); }
    else if (st.scene === 'volume') buildLayers(false);
  }

  // ---------- 3D layers (CSS 3D, one canvas per slice) ----------
  var rot = { rx: 62, rz: -34 }, spinning = false, raf = 0, sweepT = 0;
  var LAYERS = [], LCOUNT = 28, SPX = 300;
  function layerGeom() {
    var s = SPX / Math.max(V.ext[0], V.ext[1]);
    return { w: V.ext[0] * s, h: V.ext[1] * s, zt: V.ext[2] * s * parseFloat($('explode').value) };
  }
  function buildLayers(onlyHighlight) {
    var cube = $('cube'), L = Math.min(V.nz, LCOUNT), G = layerGeom(), K = V.petK;
    $('cut').max = L; if (parseInt($('cut').value, 10) > L || !onlyHighlight && LAYERS.length !== L) $('cut').value = L;
    var curK = idxOf()[2];
    if (onlyHighlight && LAYERS.length === L) {
      LAYERS.forEach(function (o) { o.el.classList.toggle('cur', Math.abs(o.k - curK) === Math.min.apply(null, LAYERS.map(function (q) { return Math.abs(q.k - curK); }))); });
      return;
    }
    cube.innerHTML = ''; LAYERS = [];
    var showPet = $('t-pet').checked && V.hasPet, showTum = $('t-tum').checked && V.hasLab, vox = $('t-vox').checked;
    for (var li = 0; li < L; li++) {
      var kk = L === 1 ? 0 : Math.round(li * (V.nz - 1) / (L - 1)), k = V.flip[2] ? kk : V.nz - 1 - kk;   // layer 0 is the feet end
      var el = document.createElement('div'); el.className = 'layer';
      var cv = document.createElement('canvas'); cv.width = V.nx; cv.height = V.ny; cv.style.imageRendering = vox ? 'pixelated' : 'auto';
      var ctx = cv.getContext('2d'), img = ctx.createImageData(V.nx, V.ny);
      for (var j = 0; j < V.ny; j++) for (var i = 0; i < V.nx; i++) {
        // far edge of the tilted layer is posterior, near edge anterior: the head faces the viewer, not mirrored
        var jj = V.flip[1] ? j : V.ny - 1 - j, ii = V.flip[0] ? V.nx - 1 - i : i, id = ii + V.nx * (jj + V.ny * k), hu = V.ct[id];
        var g = clamp((hu + 160) / 400, 0, 1) * 255, al = Math.pow(clamp((hu + 350) / 650, 0, 1), 1.3) * 0.5 + (hu > 250 ? 0.3 : 0);
        var R = g, Gc = g, B = g;
        if (showPet) {
          var pa = clamp((V.pet[id] - 2 * K) / (6 * K), 0, 1) * 0.9;
          if (pa > 0) { var hv = hot(V.pet[id] / (11 * K)); R = g * (1 - pa) + hv[0] * 255 * pa; Gc = g * (1 - pa) + hv[1] * 255 * pa; B = g * (1 - pa) + hv[2] * 255 * pa; al = Math.max(al, pa); }
        }
        if (showTum && V.lab[id] > 0) { var tc = V.lab[id] === 1 ? [255, 70, 70] : [75, 227, 154]; R = R * 0.3 + tc[0] * 0.7; Gc = Gc * 0.3 + tc[1] * 0.7; B = B * 0.3 + tc[2] * 0.7; al = Math.max(al, 0.85); }
        var q = (j * V.nx + i) * 4; img.data[q] = R; img.data[q + 1] = Gc; img.data[q + 2] = B; img.data[q + 3] = clamp(al, 0, 1) * 255;
      }
      ctx.putImageData(img, 0, 0);
      el.style.width = G.w + 'px'; el.style.height = G.h + 'px'; el.style.left = (-G.w / 2) + 'px'; el.style.top = (-G.h / 2) + 'px';
      el.appendChild(cv); cube.appendChild(el); LAYERS.push({ el: el, k: k, li: li });
    }
    var best = Math.min.apply(null, LAYERS.map(function (q) { return Math.abs(q.k - curK); }));
    LAYERS.forEach(function (o) { o.el.classList.toggle('cur', Math.abs(o.k - curK) === best); });
    var floor = document.createElement('div'); floor.className = 'floor'; floor.style.cssText = 'width:' + G.w + 'px;height:' + G.h + 'px;left:' + (-G.w / 2) + 'px;top:' + (-G.h / 2) + 'px;transform:translateZ(-6px)'; cube.appendChild(floor);
    var sw = document.createElement('div'); sw.className = 'sweep'; sw.id = 'sweep'; sw.style.cssText = 'width:' + G.w + 'px;height:' + G.h + 'px;left:' + (-G.w / 2) + 'px;top:' + (-G.h / 2) + 'px;display:none'; cube.appendChild(sw);
    ['f', 'b', 'l', 'r'].forEach(function (s) { var f = document.createElement('div'); f.className = 'face'; f.dataset.s = s; cube.appendChild(f); });
    layoutLayers();
  }
  function layoutLayers() {
    var G = layerGeom(), L = LAYERS.length, cut = parseInt($('cut').value, 10), step = L > 1 ? G.zt / (L - 1) : 0, top = step * (L - 1);
    var zc = -top / 2;
    LAYERS.forEach(function (o, n) { o.el.style.transform = 'translateZ(' + (zc + n * step) + 'px)'; o.el.style.display = n >= cut ? 'none' : 'block'; });
    var fl = document.querySelector('.floor'); if (fl) fl.style.transform = 'translateZ(' + (zc - 8) + 'px)';
    Array.prototype.forEach.call(document.querySelectorAll('.face'), function (f) {
      var s = f.dataset.s, H = top + 16, W = (s === 'f' || s === 'b') ? G.w : G.h;
      f.style.width = W + 'px'; f.style.height = H + 'px'; f.style.left = (-W / 2) + 'px'; f.style.top = (-H / 2) + 'px';
      var tz = 'translateZ(' + (zc + top / 2) + 'px) ';
      f.style.transform = s === 'f' ? 'translateY(' + (G.h / 2) + 'px) ' + tz + 'rotateX(90deg)' : s === 'b' ? 'translateY(' + (-G.h / 2) + 'px) ' + tz + 'rotateX(90deg)' :
        s === 'l' ? 'translateX(' + (-G.w / 2) + 'px) ' + tz + 'rotateZ(90deg) rotateX(90deg)' : 'translateX(' + (G.w / 2) + 'px) ' + tz + 'rotateZ(90deg) rotateX(90deg)';
    });
    $('vhud').innerHTML = '<span>' + esc(V.name) + '</span><span>X <b>' + V.ext[0].toFixed(0) + ' mm</b>  Y <b>' + V.ext[1].toFixed(0) + ' mm</b>  Z <b>' + V.ext[2].toFixed(0) + ' mm</b></span>' +
      '<span>voxel <b>' + V.sx.toFixed(2) + ' × ' + V.sy.toFixed(2) + ' × ' + V.sz.toFixed(2) + ' mm</b></span><span>' + L + ' of ' + V.nz + ' slices drawn' + (cut < L ? ', ' + (L - cut) + ' cut away' : '') + '</span>' +
      (V.real ? '<span>head up, face toward you at the reset view</span>' : '');
    applyRot();
  }
  function applyRot() {
    $('cube').style.transform = 'rotateX(' + rot.rx + 'deg) rotateZ(' + rot.rz + 'deg)';
    var a = rot.rx * Math.PI / 180, b = rot.rz * Math.PI / 180, ca = Math.cos(a), sa = Math.sin(a), cb = Math.cos(b), sb = Math.sin(b);
    function proj(x, y, z) { var X = x * cb - y * sb, Y = x * sb + y * cb; return [X, Y * ca - z * sa]; }
    var axes = [['X', proj(1, 0, 0), '#ff6b6b'], ['Y', proj(0, 1, 0), '#4be39a'], ['Z', proj(0, 0, 1), '#3fd8ff']];
    $('gizmo').innerHTML = axes.map(function (A) {
      var x = A[1][0] * 32, y = A[1][1] * 32;
      return '<line x1="0" y1="0" x2="' + x + '" y2="' + y + '" stroke="' + A[2] + '" stroke-width="2"/><text x="' + (x * 1.3 + (x >= 0 ? 2 : -9)) + '" y="' + (y * 1.3 + 4) + '" fill="' + A[2] + '" font-size="11" font-family="monospace">' + A[0] + '</text>';
    }).join('');
  }
  (function dragRot() {
    var w = $('vwrap'), down = false, lx = 0, ly = 0;
    w.addEventListener('pointerdown', function (e) { down = true; lx = e.clientX; ly = e.clientY; w.classList.add('drag'); w.setPointerCapture(e.pointerId); });
    w.addEventListener('pointermove', function (e) { if (!down) return; rot.rz += (e.clientX - lx) * 0.4; rot.rx = clamp(rot.rx - (e.clientY - ly) * 0.3, 10, 88); lx = e.clientX; ly = e.clientY; applyRot(); });
    w.addEventListener('pointerup', function () { down = false; w.classList.remove('drag'); });
    w.addEventListener('contextmenu', function (e) { if (V && V.real) e.preventDefault(); });
  })();
  function tick() {
    if (!spinning) return;
    rot.rz += 0.25; sweepT = (sweepT + 0.004) % 1; applyRot();
    var sw = $('sweep'), G = layerGeom(), top = G.zt;
    if (sw) { sw.style.display = 'block'; sw.style.transform = 'translateZ(' + (-top / 2 + sweepT * top) + 'px)'; }
    raf = requestAnimationFrame(tick);
  }
  $('t-spin').addEventListener('change', function () {
    spinning = this.checked; cancelAnimationFrame(raf);
    if (spinning) tick(); else { var sw = $('sweep'); if (sw) sw.style.display = 'none'; }
  });
  $('explode').addEventListener('input', layoutLayers); $('cut').addEventListener('input', layoutLayers);
  ['t-pet', 't-tum', 't-vox'].forEach(function (id) { $(id).addEventListener('change', function () { buildLayers(false); }); });
  $('view-reset').addEventListener('click', function () { rot.rx = 62; rot.rz = -34; $('explode').value = 1.35; $('cut').value = LAYERS.length; layoutLayers(); });

  // ---------- evaluation panel ----------
  function renderEval() {
    $('eval-demo').hidden = V.real; $('eval-real').hidden = !V.real;
    if (V.real) {
      var M = V.meta;
      $('ev-missing').textContent = '<out-root>/test_predictions/<run>/' + M.patient_id + '_pred.nii.gz';
      var rows = [['Case', M.patient_id], ['Centre', M.centre], ['Split', M.split || 'not in the split file'],
        ['Native grid', M.native_shape.join(' × ')], ['Native spacing mm', M.native_spacing.map(function (v) { return v.toFixed(3); }).join(' × ')],
        ['Shown reduced', M.reduction + '× in-plane'], ['PET scale', M.pet.scale.toPrecision(6)], ['PET scale from', M.pet.scale_source]];
      if (M.is_test) rows.push(['Note', M.test_note]);
      $('ev-case').innerHTML = rows.map(function (r) { return '<span>' + esc(r[0]) + '</span><span>' + esc(r[1]) + '</span>'; }).join('');
      return;
    }
    var s = ST, C = 2 * Math.PI * 40;
    if (!s || !s.ok) { $('ev-dice').textContent = '–'; $('ring-arc').setAttribute('stroke-dashoffset', C); ['ev-vol', 'ev-pvol', 'ev-ece', 'ev-uin', 'ev-uout'].forEach(function (i) { $(i).textContent = '–'; }); return; }
    $('ev-dice').textContent = isNaN(s.dice) ? '–' : s.dice.toFixed(3);
    $('ring-arc').setAttribute('stroke-dashoffset', isNaN(s.dice) ? C : C * (1 - s.dice));
    $('ev-vs').textContent = 'vs phantom truth';
    $('ev-vol').textContent = s.hasLab ? s.vol.toFixed(1) + ' mL' : '–'; $('ev-pvol').textContent = s.pvol.toFixed(1) + ' mL';
    $('ev-ece').textContent = isNaN(s.ece) ? '–' : s.ece.toFixed(4);
    $('ev-uin').textContent = isNaN(s.uin) ? '–' : s.uin.toFixed(2); $('ev-uout').textContent = isNaN(s.uout) ? '–' : s.uout.toFixed(2);
    var cv = $('rel'), ctx = cv.getContext('2d'), W = cv.width, H = cv.height, p = 30;
    ctx.clearRect(0, 0, W, H); ctx.strokeStyle = 'rgba(76,201,240,0.25)'; ctx.strokeRect(p, 10, W - p - 10, H - p - 10);
    ctx.setLineDash([4, 4]); ctx.beginPath(); ctx.moveTo(p, H - p); ctx.lineTo(W - 10, 10); ctx.stroke(); ctx.setLineDash([]);
    function X(v) { return p + (v - 0.5) / 0.5 * (W - p - 10); } function Y(v) { return H - p - (v - 0.5) / 0.5 * (H - p - 10); }
    ctx.strokeStyle = '#3fd8ff'; ctx.fillStyle = '#3fd8ff'; ctx.lineWidth = 1.5; ctx.beginPath();
    s.rel.forEach(function (r, i) { var x = X(r[0]), y = Y(Math.max(0.5, r[1])); if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y); }); ctx.stroke();
    s.rel.forEach(function (r) { ctx.beginPath(); ctx.arc(X(r[0]), Y(Math.max(0.5, r[1])), 2.5, 0, 6.3); ctx.fill(); });
    ctx.fillStyle = '#86a3b6'; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.fillText('confidence', W / 2, H - 6);
    ctx.save(); ctx.translate(10, H / 2); ctx.rotate(-Math.PI / 2); ctx.fillText('accuracy', 0, 0); ctx.restore();
    ctx.textAlign = 'right'; ctx.fillText('0.5', p - 4, H - p + 3); ctx.fillText('1.0', p - 4, 18);
  }

  // ---------- load / swap the volume ----------
  function setVolume(v) {
    V = v; ST = stats(V);
    document.body.classList.toggle('real', V.real);
    var red = V.real && V.meta.reduction > 1 ? ' (native ' + V.meta.native_shape.join(' × ') + ' at ' + V.meta.native_spacing.map(function (x) { return x.toFixed(2); }).join(' × ') + ' mm)' : '';
    $('grid-note').textContent = V.nx + ' × ' + V.ny + ' × ' + V.nz + ' voxels at ' + V.sx.toFixed(2) + ' × ' + V.sy.toFixed(2) + ' × ' + V.sz.toFixed(2) + ' mm' + red;
    $('tag-src').textContent = V.real ? 'Real case, local only' : 'Synthetic phantom';
    $('tag-model').textContent = V.real ? 'No model output shown' : 'Demo rule, not a trained model';
    $('sp-xy').disabled = $('sp-z').disabled = V.real; $('reset-phantom').hidden = !V.real;
    $('t-pet').disabled = !V.hasPet; $('t-tum').disabled = !V.hasLab;
    $('t-tum-text').textContent = V.real ? 'Label 1 (red), label 2 (green)' : 'Tumour';
    $('pet-max-label').textContent = V.real ? 'PET window max (tenths of case scale)' : 'PET window max (SUV)';
    $('f-unc').disabled = V.real; if (V.real) $('f-unc').checked = false;
    $('f-unc-text').textContent = V.real ? 'Uncertainty (no map loaded)' : 'Uncertainty (demo rule)';
    renderEval(); updateSlice(); if (st.scene === 'volume') buildLayers(false); else { LAYERS = []; refreshScene(); }
    document.body.setAttribute('data-showing', V.name + '|' + V.nx + 'x' + V.ny + 'x' + V.nz + '|flip ' + V.flip.join(','));
    document.body.setAttribute('data-hscroll', String(document.documentElement.scrollWidth > document.documentElement.clientWidth));
  }
  function phantom() { var xy = parseFloat($('sp-xy').value), z = parseFloat($('sp-z').value); st.cur = [74, 106, 66]; CUR = null; markRow(); setVolume(gen(xy, xy, z)); }
  ['sp-xy', 'sp-z'].forEach(function (id) { $(id).addEventListener('change', phantom); });
  $('reset-phantom').addEventListener('click', function () { $('load-msg').textContent = ''; phantom(); });

  // ---------- case browser ----------
  var INDEX = null, SHOWN = [], CUR = null, loading = false;
  function fmt(v, d) { return v == null ? '–' : Number(v).toFixed(d); }
  function num(id) { var v = parseFloat($(id).value); return isNaN(v) ? null : v; }
  function filtered() {
    var q = $('f-q').value.trim().toUpperCase(), ce = $('f-centre').value, sp = $('f-split').value, a0 = num('f-xy0'), a1 = num('f-xy1'), z0 = num('f-z0'), z1 = num('f-z1'), out = $('f-out').checked;
    return INDEX.cases.filter(function (c) {
      if (q && c.patient_id.toUpperCase().indexOf(q) < 0) return false;
      if (ce && c.centre !== ce) return false;
      if (sp && c.split !== sp) return false;
      if (a0 != null && c.sx < a0) return false; if (a1 != null && c.sx > a1) return false;
      if (z0 != null && c.sz < z0) return false; if (z1 != null && c.sz > z1) return false;
      if (out && c.pet_outlier !== true) return false;
      return true;
    });
  }
  function markRow() {
    Array.prototype.forEach.call($('case-rows').querySelectorAll('tr'), function (tr) { tr.classList.toggle('sel', tr.getAttribute('data-pid') === CUR); });
    var i = SHOWN.indexOf(CUR), none = !SHOWN.length || loading;
    $('case-prev').disabled = none || i === 0; $('case-next').disabled = none || i === SHOWN.length - 1;
  }
  function renderRows() {
    if (!INDEX) return;
    var rows = filtered(); SHOWN = rows.map(function (c) { return c.patient_id; });
    $('case-rows').innerHTML = rows.map(function (c) {
      return '<tr data-pid="' + esc(c.patient_id) + '" tabindex="0"><td>' + esc(c.patient_id) + '</td><td>' + esc(c.centre) + '</td><td>' + fmt(c.sx, 3) + '</td><td>' + fmt(c.sz, 3) + '</td><td>' +
        (c.is_test ? '<span class="testmark">' + esc(INDEX.test_note) + '</span>' : esc(c.split || 'not in the split file')) + '</td><td>' + (c.pet_scale == null ? '–' : Number(c.pet_scale).toPrecision(5)) + '</td><td>' +
        (c.pet_outlier == null ? 'unknown' : c.pet_outlier ? '<span class="flag">yes</span>' : 'no') + '</td></tr>';
    }).join('');
    var k = INDEX.split_counts;
    $('idx-note').textContent = rows.length + ' of ' + INDEX.n_cases + ' cases shown. Split: ' + k.train + ' train, ' + k.validation + ' validation, ' + k.test + ' test. PET-scale outliers: ' +
      (INDEX.n_pet_outliers == null ? 'unknown' : INDEX.n_pet_outliers) + ' (' + INDEX.outlier_rule + '). PET scale: ' + INDEX.pet_note + '.';
    markRow();
  }
  function loadIndex() {
    $('idx-note').textContent = 'Reading the case index…';
    fetch('api/index').then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); }).then(function (ix) {
      INDEX = ix;
      $('f-centre').innerHTML = '<option value="">All</option>' + ix.centres.map(function (c) { return '<option>' + esc(c) + '</option>'; }).join('');
      $('idx-missing').innerHTML = ix.missing.map(function (m) { return '<div class="empty"><b>Missing file</b><code>' + esc(m.file) + '</code><span>' + esc(m.effect) + '</span></div>'; }).join('');
      renderRows();
    }).catch(function (err) { $('idx-note').textContent = 'Could not read the case index from the local server: ' + err.message; });
  }
  function loadCase(pid) {
    if (loading) return;
    loading = true; markRow();
    var msg = $('load-msg'); msg.textContent = 'Reading ' + pid + ' from the local cache…';
    fetch('api/cases/' + encodeURIComponent(pid) + '/volume').then(function (r) {
      if (!r.ok) return r.json().then(function (j) { throw new Error(j.detail || ('HTTP ' + r.status)); });
      return r.arrayBuffer();
    }).then(function (buf) {
      var v = parseVolume(buf); CUR = pid; st.cur = labelCentre(v); loading = false; setVolume(v); markRow();
      msg.textContent = 'Showing ' + pid + ' (' + v.meta.centre + ', ' + (v.meta.split || 'not in the split file') + (v.meta.is_test ? '; ' + v.meta.test_note : '') + '). Reduced ' + v.meta.reduction + '× in-plane for display; slices are native.';
    }).catch(function (err) { loading = false; markRow(); msg.textContent = 'Could not open ' + pid + ': ' + err.message; });
  }
  function stepCase(d) { var i = SHOWN.indexOf(CUR), j = i < 0 ? 0 : i + d; if (j >= 0 && j < SHOWN.length) loadCase(SHOWN[j]); }
  $('open-cases').addEventListener('click', function () {
    var d = $('drawer'); d.hidden = !d.hidden; this.setAttribute('aria-expanded', d.hidden ? 'false' : 'true');
    if (!d.hidden && !INDEX) loadIndex();
  });
  $('case-prev').addEventListener('click', function () { stepCase(-1); });
  $('case-next').addEventListener('click', function () { stepCase(1); });
  ['f-q', 'f-xy0', 'f-xy1', 'f-z0', 'f-z1'].forEach(function (id) { $(id).addEventListener('input', renderRows); });
  ['f-centre', 'f-split', 'f-out'].forEach(function (id) { $(id).addEventListener('change', renderRows); });
  $('f-clear').addEventListener('click', function () { ['f-q', 'f-xy0', 'f-xy1', 'f-z0', 'f-z1'].forEach(function (id) { $(id).value = ''; }); $('f-centre').value = ''; $('f-split').value = ''; $('f-out').checked = false; renderRows(); });
  $('case-rows').addEventListener('click', function (e) { var tr = e.target.closest('tr[data-pid]'); if (tr) loadCase(tr.getAttribute('data-pid')); });
  $('case-rows').addEventListener('keydown', function (e) { var tr = e.target.closest('tr[data-pid]'); if (tr && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); loadCase(tr.getAttribute('data-pid')); } });

  // ---------- wiring ----------
  $('slice').addEventListener('input', function () {
    var i = parseInt(this.value, 10), q = sliceInfo(st.plane);
    st.cur[q.key] = (i + 0.5) * q.sp; updateSlice(); renderFusion();
  });
  function segClick(box, attr, key) {
    box.addEventListener('click', function (e) { var b = e.target.closest('button'); if (!b) return; st[key] = b.getAttribute(attr); syncSegs(); updateSlice(); renderFusion(); });
  }
  segClick($('plane-seg'), 'data-plane', 'plane'); segClick($('mode-seg'), 'data-mode', 'mode'); segClick($('layout-seg'), 'data-layout', 'layout');
  ['pet-op', 'pet-max'].forEach(function (id) { $(id).addEventListener('input', renderFusion); });
  ['f-gt', 'f-unc'].forEach(function (id) { $(id).addEventListener('change', renderFusion); });
  [0, 1, 2].forEach(function (n) { bindPick($('cv-' + n), n); });

  var SCENES = ['volume', 'fusion', 'spacing', 'segmentation', 'layers', 'status'];
  function show(sc) {
    if (SCENES.indexOf(sc) < 0) sc = 'volume';
    st.scene = sc;
    document.querySelector('.body').classList.toggle('wide', sc === 'layers' || sc === 'status');
    SCENES.forEach(function (s) { $('sc-' + s).hidden = s !== sc; });
    Array.prototype.forEach.call($('rail').querySelectorAll('button'), function (b) { b.setAttribute('aria-pressed', b.getAttribute('data-sc') === sc ? 'true' : 'false'); });
    if (V) refreshScene();
    if (sc !== 'volume' && spinning) { spinning = false; $('t-spin').checked = false; cancelAnimationFrame(raf); }
  }
  $('rail').addEventListener('click', function (e) { var b = e.target.closest('button'); if (!b) return; var sc = b.getAttribute('data-sc'); if (location.hash !== '#' + sc) location.hash = sc; else show(sc); });
  window.addEventListener('hashchange', function () { show(location.hash.replace('#', '')); });

  var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  phantom(); show(location.hash.replace('#', '') || 'volume');
  if (!reduce && st.scene === 'volume') { $('t-spin').checked = true; spinning = true; tick(); }
  // ?case=<id> opens that case at load
  var want = new URLSearchParams(location.search).get('case');
  if (want) { spinning = false; $('t-spin').checked = false; loadCase(want); }
})();
