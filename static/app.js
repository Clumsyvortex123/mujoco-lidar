/* WebGL2 point cloud viewer for the MuJoCo 3D lidar.
 *
 * Deliberately dependency free -- no three.js, no CDN. The whole renderer is
 * two tiny shader programs plus ~70 lines of matrix maths, which keeps the
 * page working on an air-gapped machine.
 *
 * The wire format puts points in the SENSOR frame and ships the sensor pose in
 * the header. Rather than transforming 12k points on the CPU every frame, the
 * pose goes in as two uniforms and the vertex shader does it -- so a new frame
 * costs exactly one bufferData upload and nothing else.
 */
'use strict';

const HEADER_SIZE = 56;
const MAGIC = 0x434c4a4d; // 'MJLC' read as a little-endian uint32

// ---------------------------------------------------------------------------
// Matrix helpers (column major, as GL expects)
// ---------------------------------------------------------------------------

function mat4() { return new Float32Array(16); }

function perspective(out, fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2);
  out.fill(0);
  out[0] = f / aspect;
  out[5] = f;
  out[10] = (far + near) / (near - far);
  out[11] = -1;
  out[14] = (2 * far * near) / (near - far);
  return out;
}

function lookAt(out, eye, center, up) {
  let zx = eye[0] - center[0], zy = eye[1] - center[1], zz = eye[2] - center[2];
  let zl = Math.hypot(zx, zy, zz) || 1;
  zx /= zl; zy /= zl; zz /= zl;

  let xx = up[1] * zz - up[2] * zy;
  let xy = up[2] * zx - up[0] * zz;
  let xz = up[0] * zy - up[1] * zx;
  let xl = Math.hypot(xx, xy, xz) || 1;
  xx /= xl; xy /= xl; xz /= xl;

  const yx = zy * xz - zz * xy;
  const yy = zz * xx - zx * xz;
  const yz = zx * xy - zy * xx;

  out[0] = xx; out[1] = yx; out[2] = zx; out[3] = 0;
  out[4] = xy; out[5] = yy; out[6] = zy; out[7] = 0;
  out[8] = xz; out[9] = yz; out[10] = zz; out[11] = 0;
  out[12] = -(xx * eye[0] + xy * eye[1] + xz * eye[2]);
  out[13] = -(yx * eye[0] + yy * eye[1] + yz * eye[2]);
  out[14] = -(zx * eye[0] + zy * eye[1] + zz * eye[2]);
  out[15] = 1;
  return out;
}

function multiply(out, a, b) {
  for (let c = 0; c < 4; c++) {
    const b0 = b[c * 4], b1 = b[c * 4 + 1], b2 = b[c * 4 + 2], b3 = b[c * 4 + 3];
    out[c * 4]     = a[0] * b0 + a[4] * b1 + a[8]  * b2 + a[12] * b3;
    out[c * 4 + 1] = a[1] * b0 + a[5] * b1 + a[9]  * b2 + a[13] * b3;
    out[c * 4 + 2] = a[2] * b0 + a[6] * b1 + a[10] * b2 + a[14] * b3;
    out[c * 4 + 3] = a[3] * b0 + a[7] * b1 + a[11] * b2 + a[15] * b3;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Shaders
// ---------------------------------------------------------------------------

// Shared snippet: rotate v by quaternion q stored as (x, y, z, w).
const ROTQ = `
vec3 rotq(vec4 q, vec3 v) {
  return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}`;

const RAMP = `
vec3 ramp(float t) {
  t = clamp(t, 0.0, 1.0);
  // Starts at a legible steel blue rather than near black: most returns land
  // on the ground plane at t~0, and a dark low end makes the bulk of the
  // cloud invisible against the dark background.
  vec3 c0 = vec3(0.24, 0.40, 0.68);
  vec3 c1 = vec3(0.16, 0.63, 0.75);
  vec3 c2 = vec3(0.27, 0.80, 0.52);
  vec3 c3 = vec3(0.92, 0.81, 0.29);
  vec3 c4 = vec3(0.97, 0.42, 0.24);
  if (t < 0.25) return mix(c0, c1, t / 0.25);
  if (t < 0.50) return mix(c1, c2, (t - 0.25) / 0.25);
  if (t < 0.75) return mix(c2, c3, (t - 0.50) / 0.25);
  return mix(c3, c4, (t - 0.75) / 0.25);
}`;

const POINT_VS = `#version 300 es
precision highp float;
layout(location = 0) in vec3 aPos;      // sensor frame

uniform mat4  uMVP;
uniform vec3  uOrigin;                  // sensor position in world
uniform vec4  uQuat;                    // sensor orientation, (x, y, z, w)
uniform int   uColorMode;               // 0 = height, 1 = range
uniform float uPointSize;
uniform vec2  uZRange;                  // world z min/max for the height ramp
uniform float uMaxRange;                // clip + normalisation for range mode

out vec3 vColor;
out float vDrop;
${ROTQ}
${RAMP}

void main() {
  float r = length(aPos);
  vDrop = r > uMaxRange ? 1.0 : 0.0;

  vec3 world = uOrigin + rotq(uQuat, aPos);
  gl_Position = uMVP * vec4(world, 1.0);

  float t = uColorMode == 0
    ? (world.z - uZRange.x) / max(uZRange.y - uZRange.x, 1e-3)
    : r / max(uMaxRange, 1e-3);
  vColor = ramp(t);

  // Mild perspective attenuation so near returns read as dense surfaces and
  // far ones stay legible instead of collapsing into speckle.
  gl_PointSize = clamp(uPointSize * 14.0 / max(gl_Position.w, 0.6), 1.0, 9.0);
}`;

const POINT_FS = `#version 300 es
precision highp float;
in vec3 vColor;
in float vDrop;
out vec4 fragColor;

void main() {
  if (vDrop > 0.5) discard;                     // beyond the display range
  vec2 d = gl_PointCoord - vec2(0.5);
  if (dot(d, d) > 0.25) discard;                // round splats, not squares
  fragColor = vec4(vColor, 1.0);
}`;

const LINE_VS = `#version 300 es
precision highp float;
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;

uniform mat4 uMVP;
uniform vec3 uOrigin;
uniform vec4 uQuat;

out vec3 vColor;
${ROTQ}

void main() {
  gl_Position = uMVP * vec4(uOrigin + rotq(uQuat, aPos), 1.0);
  vColor = aColor;
}`;

const LINE_FS = `#version 300 es
precision highp float;
in vec3 vColor;
uniform float uAlpha;
out vec4 fragColor;
void main() { fragColor = vec4(vColor, uAlpha); }`;

// ---------------------------------------------------------------------------
// GL setup
// ---------------------------------------------------------------------------

const canvas = document.getElementById('gl');
const gl = canvas.getContext('webgl2', { antialias: true, alpha: false });
if (!gl) {
  document.body.innerHTML =
    '<p style="padding:2rem">WebGL2 is unavailable in this browser.</p>';
  throw new Error('no webgl2');
}

function compile(type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    throw new Error('shader: ' + gl.getShaderInfoLog(sh));
  }
  return sh;
}

function program(vsSrc, fsSrc) {
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl.VERTEX_SHADER, vsSrc));
  gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fsSrc));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    throw new Error('link: ' + gl.getProgramInfoLog(p));
  }
  const uniforms = {};
  const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
  for (let i = 0; i < n; i++) {
    const name = gl.getActiveUniform(p, i).name;
    uniforms[name] = gl.getUniformLocation(p, name);
  }
  return { p, u: uniforms };
}

const pointProg = program(POINT_VS, POINT_FS);
const lineProg = program(LINE_VS, LINE_FS);

// Point cloud buffer -----------------------------------------------------
const pointVao = gl.createVertexArray();
const pointVbo = gl.createBuffer();
gl.bindVertexArray(pointVao);
gl.bindBuffer(gl.ARRAY_BUFFER, pointVbo);
gl.enableVertexAttribArray(0);
gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
gl.bindVertexArray(null);
let pointCount = 0;
let vboCapacity = 0;

// Static line geometry ---------------------------------------------------
function buildGrid(extent, step) {
  const v = [];
  const dim = [0.16, 0.18, 0.22];
  const axis = 0.5;
  for (let i = -extent; i <= extent; i += step) {
    const onAxis = Math.abs(i) < 1e-6;
    const c = onAxis ? [axis, axis, axis] : dim;
    v.push(-extent, i, 0, ...c, extent, i, 0, ...c);
    v.push(i, -extent, 0, ...c, i, extent, 0, ...c);
  }
  return new Float32Array(v);
}

// Sensor gizmo: body axes in the SENSOR frame, so the same origin+quat
// uniforms that place the cloud also place this marker.
const GIZMO = new Float32Array([
  0, 0, 0, 0.95, 0.36, 0.20,   1.2, 0, 0, 0.95, 0.36, 0.20,   // +X forward
  0, 0, 0, 0.30, 0.80, 0.45,   0, 0.8, 0, 0.30, 0.80, 0.45,   // +Y left
  0, 0, 0, 0.35, 0.60, 0.95,   0, 0, 0.8, 0.35, 0.60, 0.95,   // +Z up
]);

function makeLineVao(data) {
  const vao = gl.createVertexArray();
  const vbo = gl.createBuffer();
  gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
  gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 24, 0);
  gl.enableVertexAttribArray(1);
  gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 24, 12);
  gl.bindVertexArray(null);
  return { vao, count: data.length / 6 };
}

const gridData = buildGrid(30, 3);
const gridMesh = makeLineVao(gridData);
const gizmoMesh = makeLineVao(GIZMO);

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------

const cam = {
  target: [0, 0, 1.5],
  radius: 26,
  theta: -2.3,      // azimuth, radians
  phi: 0.45,        // elevation, radians
};

const proj = mat4(), view = mat4(), mvp = mat4();

function eyePosition() {
  const cp = Math.cos(cam.phi);
  return [
    cam.target[0] + cam.radius * cp * Math.cos(cam.theta),
    cam.target[1] + cam.radius * cp * Math.sin(cam.theta),
    cam.target[2] + cam.radius * Math.sin(cam.phi),
  ];
}

let dragging = 0;   // 0 none, 1 orbit, 2 pan
let lastX = 0, lastY = 0;

canvas.addEventListener('contextmenu', (e) => e.preventDefault());

canvas.addEventListener('pointerdown', (e) => {
  dragging = (e.button === 2 || e.shiftKey) ? 2 : 1;
  lastX = e.clientX; lastY = e.clientY;
  canvas.classList.add('dragging');
  canvas.setPointerCapture(e.pointerId);
});

canvas.addEventListener('pointerup', (e) => {
  dragging = 0;
  canvas.classList.remove('dragging');
  canvas.releasePointerCapture(e.pointerId);
});

canvas.addEventListener('pointermove', (e) => {
  if (!dragging) return;
  const dx = e.clientX - lastX, dy = e.clientY - lastY;
  lastX = e.clientX; lastY = e.clientY;

  if (dragging === 1) {
    cam.theta -= dx * 0.006;
    cam.phi = Math.max(-1.5, Math.min(1.5, cam.phi + dy * 0.006));
  } else {
    // Pan across the camera's own right/up axes, scaled by zoom so the world
    // tracks the cursor at roughly 1:1 regardless of distance.
    const eye = eyePosition();
    let fx = cam.target[0] - eye[0], fy = cam.target[1] - eye[1], fz = cam.target[2] - eye[2];
    const fl = Math.hypot(fx, fy, fz) || 1; fx /= fl; fy /= fl; fz /= fl;
    // right = normalize(forward x worldUp), with worldUp = +Z.
    let rx = fy, ry = -fx, rz = 0;
    const rl = Math.hypot(rx, ry, rz) || 1; rx /= rl; ry /= rl; rz /= rl;
    const ux = ry * fz - rz * fy, uy = rz * fx - rx * fz, uz = rx * fy - ry * fx;
    const k = cam.radius * 0.0016;
    cam.target[0] += (-rx * dx + ux * dy) * k;
    cam.target[1] += (-ry * dx + uy * dy) * k;
    cam.target[2] += (-rz * dx + uz * dy) * k;
    ui.follow.checked = false;
  }
});

canvas.addEventListener('wheel', (e) => {
  e.preventDefault();
  cam.radius = Math.max(1.2, Math.min(140, cam.radius * Math.exp(e.deltaY * 0.0012)));
}, { passive: false });

function resize() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.floor(canvas.clientWidth * dpr);
  const h = Math.floor(canvas.clientHeight * dpr);
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w; canvas.height = h;
  }
}
window.addEventListener('resize', resize);

// ---------------------------------------------------------------------------
// UI state
// ---------------------------------------------------------------------------

const ui = {
  colorMode: 0,
  pointSize: 2.5,
  maxRange: 35,
  follow: document.getElementById('follow'),
  grid: document.getElementById('grid'),
};

document.querySelectorAll('#seg-color button').forEach((btn) => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('#seg-color button').forEach((b) => b.classList.remove('on'));
    btn.classList.add('on');
    ui.colorMode = parseInt(btn.dataset.mode, 10);
  });
});

const sizeInput = document.getElementById('pt-size');
sizeInput.addEventListener('input', () => {
  ui.pointSize = parseFloat(sizeInput.value);
  document.getElementById('lbl-size').textContent = ui.pointSize.toFixed(1);
});

const rangeInput = document.getElementById('max-range');
rangeInput.addEventListener('input', () => {
  ui.maxRange = parseFloat(rangeInput.value);
  document.getElementById('lbl-range').textContent = ui.maxRange;
});

// ---------------------------------------------------------------------------
// Cloud stream
// ---------------------------------------------------------------------------

const frame = {
  seq: -1,
  nPoints: 0,
  origin: [0, 0, 1.5],
  quatXYZW: [0, 0, 0, 1],
  yawDeg: 0,
  zMin: 0,
  zMax: 8,
};

function parseCloud(buf) {
  const dv = new DataView(buf);
  if (dv.getUint32(0, true) !== MAGIC) throw new Error('bad magic');

  const nPoints = dv.getUint32(8, true);
  const seq = dv.getUint32(12, true);
  const ox = dv.getFloat32(24, true);
  const oy = dv.getFloat32(28, true);
  const oz = dv.getFloat32(32, true);
  // Header stores w, x, y, z; GLSL wants (x, y, z, w).
  const qw = dv.getFloat32(36, true);
  const qx = dv.getFloat32(40, true);
  const qy = dv.getFloat32(44, true);
  const qz = dv.getFloat32(48, true);

  const points = new Float32Array(buf, HEADER_SIZE, nPoints * 3);

  frame.seq = seq;
  frame.nPoints = nPoints;
  frame.origin = [ox, oy, oz];
  frame.quatXYZW = [qx, qy, qz, qw];
  frame.yawDeg = Math.atan2(2 * (qw * qz + qx * qy),
                            1 - 2 * (qy * qy + qz * qz)) * 180 / Math.PI;

  // Height ramp bounds follow the sensor so the colouring stays informative
  // as it climbs; a fixed 0..10 ramp goes flat once you fly above the pillars.
  frame.zMin = 0;
  frame.zMax = Math.max(oz + 4, 8);

  gl.bindVertexArray(pointVao);
  gl.bindBuffer(gl.ARRAY_BUFFER, pointVbo);
  if (points.byteLength > vboCapacity) {
    // Grow with slack so a jittering return count does not reallocate the
    // buffer every single frame.
    vboCapacity = Math.ceil(points.byteLength * 1.5);
    gl.bufferData(gl.ARRAY_BUFFER, vboCapacity, gl.DYNAMIC_DRAW);
  }
  gl.bufferSubData(gl.ARRAY_BUFFER, 0, points);
  gl.bindVertexArray(null);
  pointCount = nPoints;
}

let streamOk = false;

async function pollCloud() {
  for (;;) {
    try {
      const res = await fetch('/api/cloud?since=' + frame.seq, { cache: 'no-store' });
      if (res.status === 200) {
        parseCloud(await res.arrayBuffer());
        streamOk = true;
        showError(null);
      } else if (res.status === 204) {
        await sleep(20);           // nothing new yet
      } else {
        streamOk = false;
        await sleep(400);
      }
    } catch (err) {
      streamOk = false;
      showError(String(err.message || err));
      await sleep(700);
    }
  }
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function showError(msg) {
  const el = document.getElementById('err');
  if (!msg) { el.style.display = 'none'; return; }
  el.textContent = msg;
  el.style.display = 'block';
}

// ---------------------------------------------------------------------------
// Status panel
// ---------------------------------------------------------------------------

async function pollStatus() {
  for (;;) {
    try {
      const s = await (await fetch('/api/status', { cache: 'no-store' })).json();
      const live = s.connected && s.age_s !== null && s.age_s < 1.5;
      document.getElementById('dot').className = 'dot' + (live ? ' live' : '');
      document.getElementById('s-status').textContent =
        live ? 'streaming' : (s.connected ? 'stalled' : 'waiting for sim');
      document.getElementById('s-rate').textContent =
        live ? s.rate_hz.toFixed(1) + ' Hz' : '—';
      if (s.errors) showError(s.errors + ' malformed frame(s): ' + s.last_error);
    } catch (err) {
      document.getElementById('dot').className = 'dot';
      document.getElementById('s-status').textContent = 'server down';
    }
    await sleep(500);
  }
}

function updatePanel() {
  document.getElementById('s-points').textContent =
    frame.nPoints ? frame.nPoints.toLocaleString() : '—';
  document.getElementById('s-seq').textContent =
    frame.seq >= 0 ? frame.seq : '—';
  document.getElementById('s-pos').textContent =
    frame.origin.map((v) => v.toFixed(1)).join(', ');
  document.getElementById('s-yaw').textContent = frame.yawDeg.toFixed(0) + '°';
}

// ---------------------------------------------------------------------------
// Keyboard -> zenoh
// ---------------------------------------------------------------------------

const DRIVE_KEYS = new Set(['w', 'a', 's', 'd', 'i', 'k', 'q', 'e', 'r']);
let lastSent = 0;

document.addEventListener('keydown', (e) => {
  const key = e.key.toLowerCase();
  if (!DRIVE_KEYS.has(key)) return;
  if (e.target.matches('input, textarea')) return;
  e.preventDefault();

  // Browser auto-repeat gives smooth motion while held; throttle so a long
  // hold does not queue hundreds of commands the sim then has to chew through.
  const now = performance.now();
  if (now - lastSent < 45) return;
  lastSent = now;

  fetch('/api/cmd', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ key }),
  }).catch(() => {});
});

// ---------------------------------------------------------------------------
// Render loop
// ---------------------------------------------------------------------------

gl.clearColor(0.043, 0.051, 0.067, 1);
gl.enable(gl.DEPTH_TEST);
gl.enable(gl.BLEND);
gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

function render() {
  resize();
  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

  if (ui.follow.checked) {
    // Ease toward the sensor rather than snapping, so driving reads as motion.
    for (let i = 0; i < 3; i++) {
      cam.target[i] += (frame.origin[i] - cam.target[i]) * 0.12;
    }
  }

  const aspect = canvas.width / Math.max(canvas.height, 1);
  perspective(proj, Math.PI / 4, aspect, 0.05, 400);
  lookAt(view, eyePosition(), cam.target, [0, 0, 1]);
  multiply(mvp, proj, view);

  const IDENTITY_POSE = [0, 0, 0];
  const IDENTITY_QUAT = [0, 0, 0, 1];

  // --- ground grid (world frame => identity pose uniforms) ---
  if (ui.grid.checked) {
    gl.useProgram(lineProg.p);
    gl.uniformMatrix4fv(lineProg.u.uMVP, false, mvp);
    gl.uniform3fv(lineProg.u.uOrigin, IDENTITY_POSE);
    gl.uniform4fv(lineProg.u.uQuat, IDENTITY_QUAT);
    gl.uniform1f(lineProg.u.uAlpha, 0.85);
    gl.bindVertexArray(gridMesh.vao);
    gl.drawArrays(gl.LINES, 0, gridMesh.count);
  }

  // --- point cloud ---
  if (pointCount > 0) {
    gl.useProgram(pointProg.p);
    gl.uniformMatrix4fv(pointProg.u.uMVP, false, mvp);
    gl.uniform3fv(pointProg.u.uOrigin, frame.origin);
    gl.uniform4fv(pointProg.u.uQuat, frame.quatXYZW);
    gl.uniform1i(pointProg.u.uColorMode, ui.colorMode);
    gl.uniform1f(pointProg.u.uPointSize, ui.pointSize);
    gl.uniform2f(pointProg.u.uZRange, frame.zMin, frame.zMax);
    gl.uniform1f(pointProg.u.uMaxRange, ui.maxRange);
    gl.bindVertexArray(pointVao);
    gl.drawArrays(gl.POINTS, 0, pointCount);
  }

  // --- sensor gizmo (sensor frame => real pose uniforms) ---
  gl.useProgram(lineProg.p);
  gl.uniformMatrix4fv(lineProg.u.uMVP, false, mvp);
  gl.uniform3fv(lineProg.u.uOrigin, frame.origin);
  gl.uniform4fv(lineProg.u.uQuat, frame.quatXYZW);
  gl.uniform1f(lineProg.u.uAlpha, 1.0);
  gl.bindVertexArray(gizmoMesh.vao);
  gl.drawArrays(gl.LINES, 0, gizmoMesh.count);

  gl.bindVertexArray(null);
  updatePanel();
  requestAnimationFrame(render);
}

resize();
requestAnimationFrame(render);
pollCloud();
pollStatus();
