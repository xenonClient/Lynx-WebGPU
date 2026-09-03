import { root, useEffect, useInitData, useRef, useState } from '@lynx-js/react'
import gpu, {
  GPUBufferUsage,
  GPUShaderStage,
  GPUTextureUsage,
  createImageBitmap,
  loadAsset,
  startFrameLoop,
} from '../webgpu.js'
import '../demo.css'
import '../elements.d.ts'

/**
 * Condensation — a fogged pane of glass over a vending machine of tappable bottles.
 *
 * Swipe to wipe the fog; the wiped glass stays clear for a while and then fogs back up in patches.
 * Where the fog is, a touch wipes. Where it is gone, a tap reaches the bottle beneath and picks it.
 *
 * **Everything here is inside the WebGPU spec**, so the same bundle runs unchanged on any runtime —
 * the Metal engine or Dawn. A canvas cannot read the views it covers (on the web or here), so the
 * shelf is not Lynx content beneath the canvas: the scene **draws the shelf itself**, into a texture
 * (an SDF shader plus a glyph atlas for the words), blurs that, and composites the fog over it. Touch
 * routing is plain Lynx touch events plus a JS hit test: the same wipe field that the shader turns
 * into fog decides whether a touch is a wipe (fogged) or a tap on a bottle (clear).
 *
 * The fog itself is a CPU field of "last wiped at" timestamps (one cell per 4 CSS px) that the
 * finger stamps and the shader reads: fog = smoothstep over the age, broken up with noise so it
 * regrows in patches. The grain of tiny droplets, the old tracks where water ran down, and the drops
 * themselves — each one condensing, growing, letting go and running off the pane with a trail — are
 * procedural on top, kept sparse on purpose: a dense field of distinct beads reads as holes, which is
 * exactly the discomfort real condensation does not cause.
 *
 * On entry the pane starts clear and fogs up over a few seconds. `-altMode 1` instead starts fully
 * fogged with a preset wipe that stays clear for a long time — the screenshot harness has no finger.
 */

const FIELD_CELL = 4          // CSS px per wipe-field cell
const BRUSH_RADIUS = 30       // CSS px
const BRUSH_CORE = 0.4        // the inner fraction of the brush that wipes fully
const HOLD_SECONDS = 4.0      // wiped glass stays clear this long…
const GROW_SECONDS = 7.0      // …then fogs back over this long
const PRESET_HOLD_SECONDS = 40
const RESIDUE = 0.04          // wiped glass is never perfectly clear
const NEVER = -1e5            // "never wiped" — any age beyond hold + grow is full fog
const CLEAR_THRESHOLD = 0.45  // regrow below this counts as clear glass for touch routing
const TAP_SLOP = 12           // CSS px a finger may drift and still be a tap
const STATS_INTERVAL = 8      // frames between clear-glass statistics

// ── The shelf layout (CSS px) — shared by the content shader and the JS hit test ─────────

const SHOP_TOP = 206
const SIDE_INSET = 14
const SHELF_PAD = 8
const SHELF_PITCH = 136
const SHELF_HEIGHT = 128
const BOTTLE_W = 58
const BOTTLE_H = 86
const BOTTLE_TOP = 11
const STRIP_TOP = BOTTLE_TOP + BOTTLE_H + 8
const STRIP_H = 22
const COLUMNS = 5
const ROWS = 4

interface Bottle {
  name: string
  color: string
  cap: string
  price: string
}

/** Names and prices are drawn from `fog-labels.png` (Tools/make-fog-labels.py) — keep the two lists in step. */
const SHELVES: Bottle[][] = [
  [
    { name: 'COLA', color: '#e0262d', cap: '#ff8a80', price: '1,250' },
    { name: 'SODA', color: '#2ec27e', cap: '#b9f6ca', price: '1,150' },
    { name: 'LEMON', color: '#ffd23f', cap: '#fff59d', price: '1,300' },
    { name: 'GRAPE', color: '#7b5be6', cap: '#d1c4e9', price: '1,400' },
    { name: 'CIDER', color: '#38c6ff', cap: '#b3ecff', price: '1,100' },
  ],
  [
    { name: 'PEACH', color: '#ff7a59', cap: '#ffd0c2', price: '1,500' },
    { name: 'MINT', color: '#19c9b8', cap: '#a7f3ec', price: '1,350' },
    { name: 'BERRY', color: '#ff2d78', cap: '#ffb3cf', price: '1,600' },
    { name: 'MELON', color: '#8fd63c', cap: '#dcf5b5', price: '1,250' },
    { name: 'COCOA', color: '#9c6b3f', cap: '#e0c3a3', price: '1,800' },
  ],
  [
    { name: 'ORANGE', color: '#ff9a1f', cap: '#ffe0b2', price: '1,300' },
    { name: 'LIME', color: '#b9f23a', cap: '#f0ffc2', price: '1,150' },
    { name: 'OCEAN', color: '#2b6cff', cap: '#bcd0ff', price: '1,700' },
    { name: 'PLUM', color: '#c23fb7', cap: '#f3c4ee', price: '1,450' },
    { name: 'MILK', color: '#f2f2f2', cap: '#c9c9c9', price: '1,000' },
  ],
  [
    { name: 'CHERRY', color: '#ff4b4b', cap: '#ffc9c9', price: '1,350' },
    { name: 'KIWI', color: '#5dbb46', cap: '#c8f0bf', price: '1,250' },
    { name: 'SKY', color: '#57b8ff', cap: '#cfeaff', price: '1,200' },
    { name: 'HONEY', color: '#f5b300', cap: '#ffe9a6', price: '1,550' },
    { name: 'VIOLET', color: '#9147ff', cap: '#dcc6ff', price: '1,650' },
  ],
]

const ALL_BOTTLES: Bottle[] = ([] as Bottle[]).concat(...SHELVES)

function hexToRgb(hex: string): [number, number, number] {
  const value = parseInt(hex.slice(1), 16)
  return [((value >> 16) & 255) / 255, ((value >> 8) & 255) / 255, (value & 255) / 255]
}

/** `toLocaleString` is not something to lean on in PrimJS — thousands separators by hand. */
function formatWon(amount: number) {
  return `₩${String(amount).replace(/\B(?=(\d{3})+(?!\d))/g, ',')}`
}

/** The bottle under a CSS point, or -1. The same geometry the content shader draws. */
function bottleAt(x: number, y: number, cssWidth: number) {
  const row = Math.floor((y - SHOP_TOP) / SHELF_PITCH)
  if (row < 0 || row >= ROWS) return -1
  const top = SHOP_TOP + row * SHELF_PITCH + BOTTLE_TOP
  if (y < top || y > top + BOTTLE_H) return -1
  const innerLeft = SIDE_INSET + SHELF_PAD
  const spacing = (cssWidth - 2 * innerLeft - BOTTLE_W) / (COLUMNS - 1)
  for (let column = 0; column < COLUMNS; column++) {
    const left = innerLeft + column * spacing
    if (x >= left && x <= left + BOTTLE_W) return row * COLUMNS + column
  }
  return -1
}

/** A preset wipe for the screenshot harness — an S-swipe and a rubbed circle, in fractions of the canvas. */
const PRESET_STROKES: number[][][] = [
  [[0.12, 0.30], [0.50, 0.27], [0.86, 0.33], [0.58, 0.44], [0.20, 0.49], [0.55, 0.57], [0.85, 0.62]],
  [[0.30, 0.76], [0.42, 0.72], [0.56, 0.76], [0.58, 0.83], [0.44, 0.87], [0.32, 0.83], [0.30, 0.76], [0.45, 0.80], [0.56, 0.76]],
]

// ── Shaders ─────────────────────────────────────────────────────────

/** The shelf, drawn from the layout above — rounded boxes and the glyph atlas. Rendered into a texture. */
const SHELF_SHADER = /* wgsl */ `
struct Shop {
  resolution: vec2f,
  scale: f32,
  cssWidth: f32,
  colors: array<vec4f, 20>,
  caps: array<vec4f, 20>,
  picked: array<vec4f, 5>,
};
@group(0) @binding(0) var<uniform> shop: Shop;
@group(0) @binding(1) var labels: texture_2d<f32>;
@group(0) @binding(2) var labelSampler: sampler;

const SHOP_TOP = ${SHOP_TOP}.0;
const SIDE_INSET = ${SIDE_INSET}.0;
const SHELF_PAD = ${SHELF_PAD}.0;
const SHELF_PITCH = ${SHELF_PITCH}.0;
const SHELF_HEIGHT = ${SHELF_HEIGHT}.0;
const BOTTLE_W = ${BOTTLE_W}.0;
const BOTTLE_H = ${BOTTLE_H}.0;
const BOTTLE_TOP = ${BOTTLE_TOP}.0;
const STRIP_TOP = ${STRIP_TOP}.0;
const STRIP_H = ${STRIP_H}.0;

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
  var corners = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  return vec4f(corners[index], 0.0, 1.0);
}

fn rounded_box(point: vec2f, half: vec2f, radius: f32) -> f32 {
  let q = abs(point) - half + vec2f(radius, radius);
  return length(max(q, vec2f(0.0, 0.0))) + min(max(q.x, q.y), 0.0) - radius;
}

fn fill(distance: f32, softness: f32) -> f32 {
  return 1.0 - smoothstep(-softness, softness, distance);
}

fn segment(point: vec2f, a: vec2f, b: vec2f) -> f32 {
  let pa = point - a;
  let ba = b - a;
  let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
  return length(pa - ba * h);
}

/// The alpha of the glyph atlas cell (column, row) at the box-relative position (0~1).
fn glyph(cell: vec2f, inside: vec2f) -> f32 {
  let uv = (cell + clamp(inside, vec2f(0.0, 0.0), vec2f(1.0, 1.0))) / vec2f(5.0, 8.0);
  return textureSampleLevel(labels, labelSampler, uv, 0.0).a;
}

@fragment
fn fs_main(@builtin(position) fragment: vec4f) -> @location(0) vec4f {
  let c = fragment.xy / shop.scale;   // CSS px, y down
  let width = shop.cssWidth;
  var color = vec3f(0.024, 0.031, 0.055);

  let shelfIndex = floor((c.y - SHOP_TOP) / SHELF_PITCH);
  if (c.y >= SHOP_TOP && shelfIndex < 4.0) {
    let top = SHOP_TOP + shelfIndex * SHELF_PITCH;

    // The shelf panel with its 1px border, and the LED price strip along its bottom.
    let panel = rounded_box(c - vec2f(width * 0.5, top + SHELF_HEIGHT * 0.5),
                            vec2f(width * 0.5 - SIDE_INSET, SHELF_HEIGHT * 0.5), 10.0);
    color = mix(color, vec3f(0.122, 0.153, 0.220), fill(panel, 0.5));
    color = mix(color, vec3f(0.063, 0.078, 0.122), fill(panel + 1.0, 0.5));
    let strip = rounded_box(c - vec2f(width * 0.5, top + STRIP_TOP + STRIP_H * 0.5),
                            vec2f(width * 0.5 - SIDE_INSET - 1.0, STRIP_H * 0.5), 6.0);
    color = mix(color, vec3f(0.094, 0.114, 0.169), fill(strip, 0.5));

    // Which slot this pixel belongs to, and that bottle's data.
    let innerLeft = SIDE_INSET + SHELF_PAD;
    let spacing = (width - 2.0 * innerLeft - BOTTLE_W) / 4.0;
    let slot = clamp(floor((c.x - innerLeft) / spacing + 0.5 - BOTTLE_W * 0.5 / spacing), 0.0, 4.0);
    let index = i32(shelfIndex) * 5 + i32(slot);
    let picked = shop.picked[index / 4][index % 4];
    let grow = 1.0 + 0.07 * picked;
    let center = vec2f(innerLeft + slot * spacing + BOTTLE_W * 0.5, top + BOTTLE_TOP + BOTTLE_H * 0.5);
    let local = (c - center) / grow;

    // The bottle: a white ring when picked, else a faint one.
    let body = rounded_box(local, vec2f(BOTTLE_W * 0.5, BOTTLE_H * 0.5), 14.0);
    let ring = mix(mix(color, vec3f(1.0, 1.0, 1.0), 0.12), vec3f(1.0, 1.0, 1.0), picked);
    color = mix(color, ring, fill(body, 0.6));
    color = mix(color, shop.colors[index].rgb, fill(body + 2.0, 0.6));

    // The cap, the label with its name, the check mark of a picked bottle.
    let cap = rounded_box(local - vec2f(0.0, -BOTTLE_H * 0.5 + 5.0 + 6.0), vec2f(13.0, 6.0), 4.0);
    color = mix(color, shop.caps[index].rgb, fill(cap, 0.6));
    let labelCenter = vec2f(0.0, -BOTTLE_H * 0.5 + 29.0 + 11.0);
    let label = rounded_box(local - labelCenter, vec2f(22.0, 11.0), 6.0);
    color = mix(color, vec3f(0.92, 0.92, 0.92), fill(label, 0.6));
    let inside = (local - labelCenter + vec2f(22.0, 11.0)) / vec2f(44.0, 22.0);
    let name = glyph(vec2f(slot, shelfIndex), inside) * fill(label, 0.6);
    color = mix(color, vec3f(0.10, 0.10, 0.10), name);
    let checkCenter = vec2f(0.0, -BOTTLE_H * 0.5 + 29.0 + 22.0 + 11.0);
    let check = min(segment(local - checkCenter, vec2f(-7.0, 0.0), vec2f(-2.0, 5.0)),
                    segment(local - checkCenter, vec2f(-2.0, 5.0), vec2f(8.0, -6.0)));
    color = mix(color, vec3f(1.0, 1.0, 1.0), fill(check - 1.4, 0.6) * picked);

    // The price on the strip — the second half of the atlas.
    let priceCenter = vec2f(center.x, top + STRIP_TOP + STRIP_H * 0.5);
    let priceInside = (c - priceCenter + vec2f(22.0, 11.0)) / vec2f(44.0, 22.0);
    let onStrip = fill(strip, 0.5) * step(0.0, priceInside.x) * step(priceInside.x, 1.0);
    let price = glyph(vec2f(slot, shelfIndex + 4.0), priceInside) * onStrip;
    color = mix(color, vec3f(0.427, 0.882, 0.627), price);
  }
  return vec4f(color, 1.0);
}
`

/** Separable Gaussian, run several times each way on a half-size copy of the shelf — the milk of the fog. */
const BLUR_SHADER = /* wgsl */ `
struct BlurUniforms {
  direction: vec2f,
  texel: vec2f,
};
@group(0) @binding(0) var<uniform> b: BlurUniforms;
@group(0) @binding(1) var source: texture_2d<f32>;
@group(0) @binding(2) var blurSampler: sampler;

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOutput {
  var corners = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  let corner = corners[index];
  var out: VertexOutput;
  out.position = vec4f(corner, 0.0, 1.0);
  // Texture rows run top to bottom, NDC runs bottom to top.
  out.uv = vec2f(corner.x * 0.5 + 0.5, 0.5 - corner.y * 0.5);
  return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
  let offset = b.direction * b.texel;
  var color = textureSample(source, blurSampler, in.uv) * 0.2270270270;
  color = color + textureSample(source, blurSampler, in.uv + offset * 1.3846153846) * 0.3162162162;
  color = color + textureSample(source, blurSampler, in.uv - offset * 1.3846153846) * 0.3162162162;
  color = color + textureSample(source, blurSampler, in.uv + offset * 3.2307692308) * 0.0702702703;
  color = color + textureSample(source, blurSampler, in.uv - offset * 3.2307692308) * 0.0702702703;
  return color;
}
`

/**
 * The glass — fog from the wipe field, grain, tracks, running drops, the rim of water at the wipe's edge.
 *
 * Identifier hygiene matters here: WGSL reserves words such as `active`, `patch`, `pass`, `filter`,
 * `layout` and `set`, and a spec-conformant compiler (Dawn's Tint) refuses the whole module over one
 * of them — the bundled transpiler is lenient, so the mistake only shows on the other runtime.
 */
const GLASS_SHADER = /* wgsl */ `
struct Uniforms {
  resolution: vec2f,
  grid: vec2f,
  time: f32,
  hold: f32,
  grow: f32,
  residue: f32,
};
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var sharpTex: texture_2d<f32>;
@group(0) @binding(2) var blurTex: texture_2d<f32>;
@group(0) @binding(3) var glassSampler: sampler;
@group(0) @binding(4) var wipeTex: texture_2d<f32>;

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
  var corners = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  return vec4f(corners[index], 0.0, 1.0);
}

fn hash21(p: vec2f) -> f32 {
  var q = fract(p * vec2f(123.34, 456.21));
  q = q + vec2f(dot(q, q + vec2f(45.32, 45.32)));
  return fract(q.x * q.y);
}

fn hash22(p: vec2f) -> vec2f {
  let h = hash21(p);
  return vec2f(h, hash21(p + vec2f(h, 17.31)));
}

fn value_noise(p: vec2f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let w = f * f * (vec2f(3.0, 3.0) - 2.0 * f);
  let a = hash21(i);
  let b = hash21(i + vec2f(1.0, 0.0));
  let c = hash21(i + vec2f(0.0, 1.0));
  let d = hash21(i + vec2f(1.0, 1.0));
  return mix(mix(a, b, w.x), mix(c, d, w.x), w.y);
}

fn fbm(p: vec2f) -> f32 {
  var sum = 0.0;
  var amplitude = 0.5;
  var q = p;
  for (var i = 0; i < 3; i = i + 1) {
    sum = sum + value_noise(q) * amplitude;
    q = q * 2.03 + vec2f(1.7, 9.2);
    amplitude = amplitude * 0.5;
  }
  return sum;
}

/// The wipe field is far coarser than the screen — bilinear over textureLoad keeps it smooth. The
/// texture is r32float, bound as unfilterable-float (a sampler could not touch it on every runtime).
fn wiped_at(uv: vec2f) -> f32 {
  let g = uv * u.grid - vec2f(0.5, 0.5);
  let base = floor(g);
  let f = fract(g);
  let last = vec2i(u.grid) - vec2i(1, 1);
  let i0 = clamp(vec2i(base), vec2i(0, 0), last);
  let i1 = clamp(vec2i(base) + vec2i(1, 1), vec2i(0, 0), last);
  let a = textureLoad(wipeTex, vec2i(i0.x, i0.y), 0).r;
  let b = textureLoad(wipeTex, vec2i(i1.x, i0.y), 0).r;
  let c = textureLoad(wipeTex, vec2i(i0.x, i1.y), 0).r;
  let d = textureLoad(wipeTex, vec2i(i1.x, i1.y), 0).r;
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

/// The grain of condensation — droplets far too small to read as objects, so they are relief, not
/// beads: each catches the light on the side facing it and shadows the other. Returns (mask, shade).
fn grain(p: vec2f, cells: f32, seed: f32) -> vec2f {
  let g = p * cells;
  let id = floor(g);
  let f = fract(g) - vec2f(0.5, 0.5);
  let r = hash22(id + vec2f(seed, seed));
  let center = (r - vec2f(0.5, 0.5)) * 0.7;
  let size = 0.10 + r.x * 0.16;
  let presence = step(0.5, hash21(id + vec2f(3.1, seed)));
  let d = (f - center) / size;
  let mask = smoothstep(1.0, 0.35, length(d)) * presence;
  let shade = dot(normalize(d + vec2f(0.0001, 0.0001)), vec2f(-0.6, -0.8));
  return vec2f(mask, shade);
}

/// What one drop looks like against the pane. q is the offset from its centre in radii.
///
/// A drop on fogged glass is the same milky pane, only **sharper** where it sits: the fog colour
/// plus whatever detail the blur had taken away, inverted by the lens. Over a plain backdrop that
/// leaves nothing but the rim and the glint — which is how real drops read — and never a dark bead.
fn shade_drop(uv: vec2f, q: vec2f, radiusUV: vec2f, milk: vec3f, blurred: vec3f) -> vec3f {
  let rr = length(q);
  let rim = smoothstep(0.45, 1.0, rr);
  let nz = sqrt(max(1.0 - dot(q, q), 0.0));
  let normal = normalize(vec3f(q.x, q.y, nz * 1.3));
  let light = normalize(vec3f(-0.45, -0.6, 0.66));
  let seen = textureSample(sharpTex, glassSampler, uv - q * radiusUV * 2.2).rgb;
  var color = milk + (seen - blurred) * 0.9;
  color = color * (1.0 - rim * 0.22);
  let ring = smoothstep(0.62, 0.9, rr) * (1.0 - smoothstep(0.9, 1.0, rr));
  color = color + vec3f(0.14, 0.15, 0.17) * ring;
  let spec = pow(max(dot(normal, light), 0.0), 40.0);
  color = color + vec3f(0.95, 0.97, 1.0) * spec * 0.8;
  let focus = smoothstep(0.15, 0.9, dot(q, vec2f(0.45, 0.85))) * (1.0 - rim) * 0.14;
  return color + vec3f(focus, focus, focus);
}

/// Old tracks — where drops ran down some time ago and the condensation never fully closed again.
/// Static, a few px wide with soft edges, each starting and ending somewhere on the pane.
fn tracks(p: vec2f, columns: f32, seed: f32) -> f32 {
  let column = floor(p.x * columns);
  let r = hash21(vec2f(column, seed));
  let alive = step(0.62, r);
  let wobble = sin(p.y * 28.0 + r * 40.0) * 0.0025 + sin(p.y * 71.0 + r * 9.0) * 0.001;
  let centerX = (column + 0.5) / columns + (r - 0.5) * 0.6 / columns + wobble;
  let dx = abs(p.x - centerX);
  let top = r * 0.55;
  let bottom = top + 0.25 + hash21(vec2f(column, seed + 1.0)) * 0.7;
  let along = smoothstep(top, top + 0.06, p.y) * (1.0 - smoothstep(bottom - 0.08, bottom, p.y));
  let width = 0.003 + hash21(vec2f(column, seed + 2.0)) * 0.003;
  return smoothstep(width, width * 0.35, dx) * along * alive;
}

/// One run of one column: a drop condenses somewhere in the upper half, grows, sits, then lets go
/// and runs down the pane — accelerating, wobbling, off the bottom — leaving a wet trail as wide as
/// itself. Returns (head mask, head qx, head qy, trail).
fn drip_run(p: vec2f, columns: f32, seed: f32, column: f32, cycle: f32, phase: f32, presence: f32) -> vec4f {
  let rc = hash22(vec2f(column, seed) + vec2f(cycle * 1.7, cycle * 3.1));
  let alive = step(presence, rc.x);
  let x0 = (column + 0.5) / columns + (rc.y - 0.5) * 0.6 / columns;
  let y0 = 0.03 + hash21(vec2f(column + cycle, seed + 5.0)) * 0.55;
  let fullSize = 0.0045 + hash21(vec2f(column - cycle, seed + 9.0)) * 0.004;
  let grow = smoothstep(0.0, 0.18, phase);
  let fallT = clamp((phase - 0.6) / 0.4, 0.0, 1.0);
  let falling = step(0.0001, fallT);
  let y = y0 + fallT * fallT * 1.3;
  let x = x0 + sin(p.y * 40.0 + rc.x * 20.0) * 0.0025 * falling;
  let size = fullSize * (0.45 + 0.55 * grow);
  var d = p - vec2f(x, y);
  d.y = d.y * mix(0.86, 0.68, falling);
  let q = d / size;
  let head = (1.0 - smoothstep(0.88, 1.05, length(q))) * grow;
  let inTrail = step(y0, p.y) * step(p.y, y);
  let along = (p.y - y0) / max(y - y0, 0.0001);
  let trail = smoothstep(size * 0.75, size * 0.25, abs(p.x - x)) * inTrail * (0.3 + 0.7 * along) * falling;
  return vec4f(head, q.x, q.y, trail) * alive;
}

/// A layer of columns, each running its own clock. The previous run's trail lingers into the next
/// cycle and fades as the film closes over it. Returns (head, qx, qy, trail).
fn drips(p: vec2f, t: f32, columns: f32, seed: f32, presence: f32) -> vec4f {
  let column = floor(p.x * columns);
  let r = hash21(vec2f(column, seed));
  let period = 16.0 + r * 22.0;
  let clock = t / period + r * 9.0;
  let cycle = floor(clock);
  let phase = fract(clock);
  let now = drip_run(p, columns, seed, column, cycle, phase, presence);
  let before = drip_run(p, columns, seed, column, cycle - 1.0, 1.0, presence);
  let linger = 1.0 - smoothstep(0.0, 0.55, phase);
  return vec4f(now.x, now.y, now.z, max(now.w, before.w * linger));
}

@fragment
fn fs_main(@builtin(position) fragment: vec4f) -> @location(0) vec4f {
  let uv = fragment.xy / u.resolution;
  let aspect = u.resolution.x / u.resolution.y;
  let p = vec2f(uv.x * aspect, uv.y);
  let sharp = textureSample(sharpTex, glassSampler, uv).rgb;

  // 1) How much fog is here — regrowth since the last wipe, broken up so it comes back in patches.
  //    The lookup is warped by noise so even a ruler-straight swipe leaves a ragged edge.
  let warp = (vec2f(fbm(p * 7.0 + vec2f(2.0, 5.0)), fbm(p * 7.0 + vec2f(41.0, 13.0))) - vec2f(0.44, 0.44)) * 0.035;
  let age = u.time - wiped_at(uv + warp);
  let regrow = smoothstep(u.hold, u.hold + u.grow, age);
  let patches = fbm(p * 5.0 + vec2f(0.0, u.time * 0.015));
  let wipeFog = smoothstep(patches * 0.6, patches * 0.6 + 0.4, regrow);

  // 2) Water on the move — every drop is a run: it condenses, grows, lets go and runs off the pane,
  //    and its trail (as wide as the drop) lingers. Trails and the older tracks only thin the fog:
  //    a wet streak is a hazier window in real condensation, not a hole.
  let dripA = drips(p, u.time, 9.0, 1.0, 0.62);
  let dripB = drips(p + vec2f(0.037, 0.0), u.time, 16.0, 7.0, 0.66);
  let dripC = drips(p + vec2f(0.011, 0.0), u.time, 26.0, 13.0, 0.72);
  let head = max(dripA.x, max(dripB.x, dripC.x));
  let headQ = select(select(dripA.yz, dripB.yz, dripB.x > dripA.x), dripC.yz, dripC.x > max(dripA.x, dripB.x));
  let trail = max(dripA.w, max(dripB.w, dripC.w));
  let track = max(tracks(p, 12.0, 1.0), tracks(p + vec2f(0.013, 0.0), 20.0, 5.0) * 0.8);
  let streak = max(track * 0.8, trail);
  let fog = wipeFog * (1.0 - trail * 0.28);

  // 3) The fogged glass. Condensation is a scattering film, not a sheet laid over the scene: what is
  //    bright behind it blooms toward white, what is dark stays a dim grey, and the film is thicker
  //    here and thinner there in slow patches. No flat veil.
  let blurred = textureSample(blurTex, glassSampler, uv).rgb;
  let density = 0.55 + 0.45 * fbm(p * 2.6 + vec2f(1.0, u.time * 0.008));
  let fine = (value_noise(p * 260.0) - 0.5) * 0.02;
  let glow = blurred * (0.85 + 0.25 * density);
  let luma = dot(glow, vec3f(0.30, 0.59, 0.11));
  var color = glow + vec3f(luma * 0.2, luma * 0.2, luma * 0.22) + vec3f(0.15, 0.16, 0.185) * density + vec3f(fine, fine, fine);
  //    A streak shows the shelf a little sharper through a film of water — inside the fog colour.
  color = mix(color, mix(sharp, blurred, 0.4) * 0.9 + vec3f(0.07, 0.075, 0.085), streak * 0.5 * wipeFog);

  // 4) The grain of condensation — tiny droplets as relief: lit on the side facing the light, shadowed
  //    on the other. Sparse, and never a bead.
  let grainA = grain(p, 210.0, 0.0);
  let grainB = grain(p, 140.0, 4.0);
  let relief = (0.04 + 0.06 * grainA.y) * grainA.x * 0.6 + (0.04 + 0.06 * grainB.y) * grainB.x * 0.4;
  color = color + vec3f(relief, relief, relief * 1.05) * fog;

  // 5) The drops themselves — lenses over the sharp shelf, sitting or running. They belong to the
  //    fog: a wipe takes them with it, and they condense again as it returns.
  let milk = color;
  let headColor = shade_drop(uv, headQ, vec2f(0.0062 / aspect, 0.0062), milk, blurred);
  color = mix(color, headColor, head * wipeFog);

  // 6) The rim — water the finger pushed to the edge of the wipe. Only where the fog actually has an
  //    edge: the gradient gates it, so a pane fogging up as a whole grows no rim.
  let gradient = vec2f(dpdx(regrow), dpdy(regrow));
  let edge = clamp(length(gradient) * 40.0, 0.0, 1.0);
  let rim = smoothstep(0.03, 0.30, wipeFog) * (1.0 - smoothstep(0.30, 0.80, wipeFog)) * edge;
  let rimDirection = gradient / max(length(gradient), 0.00001);
  let rimColor = textureSample(sharpTex, glassSampler, uv + rimDirection * 0.02 * rim).rgb;
  color = mix(color, rimColor * 1.15 + vec3f(0.10, 0.10, 0.10), rim * 0.75);

  var alpha = max(fog, rim * 0.85);
  alpha = max(alpha, head * 0.7 * wipeFog);
  alpha = max(alpha, u.residue);
  return vec4f(mix(sharp, color, alpha), 1.0);
}
`

// ── The wipe field ──────────────────────────────────────────────────

function smoothstep(edge0: number, edge1: number, x: number) {
  const t = Math.min(Math.max((x - edge0) / (edge1 - edge0), 0), 1)
  return t * t * (3 - 2 * t)
}

/**
 * "Last wiped at" per cell, in seconds of scene time. The shader turns age into fog; touch routing
 * reads the same numbers, so what looks clear and what counts as clear never disagree.
 */
class WipeField {
  readonly columns: number
  readonly rows: number
  readonly values: Float32Array
  dirty = true

  constructor(cssWidth: number, cssHeight: number) {
    this.columns = Math.max(1, Math.ceil(cssWidth / FIELD_CELL))
    this.rows = Math.max(1, Math.ceil(cssHeight / FIELD_CELL))
    this.values = new Float32Array(this.columns * this.rows).fill(NEVER)
  }

  /** Wipes along the segment (ax, ay) → (bx, by), CSS px. The brush core wipes fully, the edge only partly. */
  stamp(ax: number, ay: number, bx: number, by: number, now: number, hold: number, grow: number) {
    const total = hold + grow
    const dx = bx - ax
    const dy = by - ay
    const lengthSquared = dx * dx + dy * dy
    const minColumn = Math.max(0, Math.floor((Math.min(ax, bx) - BRUSH_RADIUS) / FIELD_CELL))
    const maxColumn = Math.min(this.columns - 1, Math.ceil((Math.max(ax, bx) + BRUSH_RADIUS) / FIELD_CELL))
    const minRow = Math.max(0, Math.floor((Math.min(ay, by) - BRUSH_RADIUS) / FIELD_CELL))
    const maxRow = Math.min(this.rows - 1, Math.ceil((Math.max(ay, by) + BRUSH_RADIUS) / FIELD_CELL))
    for (let row = minRow; row <= maxRow; row++) {
      const py = (row + 0.5) * FIELD_CELL
      for (let column = minColumn; column <= maxColumn; column++) {
        const px = (column + 0.5) * FIELD_CELL
        let t = lengthSquared > 0 ? ((px - ax) * dx + (py - ay) * dy) / lengthSquared : 0
        t = Math.min(Math.max(t, 0), 1)
        const ex = ax + t * dx - px
        const ey = ay + t * dy - py
        const distance = Math.sqrt(ex * ex + ey * ey)
        if (distance >= BRUSH_RADIUS) continue
        // Cells at the brush edge are stamped as if wiped a while ago — they fog back first.
        const value = now - total * smoothstep(BRUSH_RADIUS * BRUSH_CORE, BRUSH_RADIUS, distance)
        const index = row * this.columns + column
        if (value > this.values[index]) {
          this.values[index] = value
          this.dirty = true
        }
      }
    }
  }

  /** 0 = just wiped, 1 = full fog — the same curve the shader uses, at the cell under a CSS point. */
  regrowAt(x: number, y: number, now: number, hold: number, grow: number) {
    const column = Math.min(this.columns - 1, Math.max(0, Math.floor(x / FIELD_CELL)))
    const row = Math.min(this.rows - 1, Math.max(0, Math.floor(y / FIELD_CELL)))
    return smoothstep(hold, hold + grow, now - this.values[row * this.columns + column])
  }

  /** The fraction of cells that count as clear glass — for the HUD. */
  clearFraction(now: number, hold: number, grow: number) {
    let clear = 0
    for (let i = 0; i < this.values.length; i += 4) {
      if (smoothstep(hold, hold + grow, now - this.values[i]) < CLEAR_THRESHOLD) clear += 1
    }
    return clear / Math.ceil(this.values.length / 4)
  }

  fill(value: number) {
    this.values.fill(value)
    this.dirty = true
  }
}

interface Resources {
  cssWidth: number
  cssHeight: number
  pixelWidth: number
  pixelHeight: number
  field: WipeField
  wipeTexture: any
  shelfTexture: any
  blurA: any
  blurB: any
  blurBuffers: any[]
  blurPasses: { bindGroup: any; target: any }[]
  shelfBindGroup: any
  glassBindGroup: any
  shelfDirty: boolean
}

interface Gesture {
  /** Whether the touch began on clear glass — then it is a tap on the shelf, not a wipe. */
  onShelf: boolean
  startX: number
  startY: number
  moved: boolean
}

function FogScene() {
  const [fps, setFps] = useState(0)
  const [clock, setClock] = useState(0)
  const [status, setStatus] = useState('')
  const [picks, setPicks] = useState<number[]>([])
  const [lastInput, setLastInput] = useState('none yet — wipe the glass to reach the shelf')
  const [clearPercent, setClearPercent] = useState(0)
  const initData = useInitData() as { altMode?: boolean } | undefined
  const preset = useRef(!!initData?.altMode)

  const canvasCss = useRef({ width: 0, height: 0 })
  const stroke = useRef<{ x: number; y: number } | null>(null)
  const gesture = useRef<Gesture | null>(null)
  /** Segments queued for the frame loop — [ax, ay, bx, by, …] in CSS px. */
  const pendingSegments = useRef<number[]>([])
  /** What the frame loop reads — routing needs the field, which lives in the loop. */
  const routing = useRef<{ isClear: (x: number, y: number) => boolean } | null>(null)
  const picksRef = useRef<number[]>([])
  const commands = useRef({ wipeAll: false, fogAll: false, shelfDirty: false })

  function touchPoint(event: any) {
    const touch = event?.touches?.[0] ?? event?.changedTouches?.[0]
    if (!touch) return null
    const { width, height } = canvasCss.current
    return {
      x: Math.min(Math.max(touch.x, 0), Math.max(width, 1)),
      y: Math.min(Math.max(touch.y, 0), Math.max(height, 1)),
    }
  }

  function togglePick(index: number) {
    const bottle = ALL_BOTTLES[index]
    const picked = picksRef.current.includes(index)
    picksRef.current = picked
      ? picksRef.current.filter((value: number) => value !== index)
      : [...picksRef.current, index]
    setPicks(picksRef.current)
    setLastInput(`shelf → ${bottle.name} ${picked ? 'put back' : 'picked'} (through clear glass)`)
    commands.current.shelfDirty = true
  }

  function onTouchStart(event: any) {
    const point = touchPoint(event)
    if (!point) return
    const onShelf = routing.current ? routing.current.isClear(point.x, point.y) : false
    gesture.current = { onShelf, startX: point.x, startY: point.y, moved: false }
    if (onShelf) return
    stroke.current = point
    pendingSegments.current.push(point.x, point.y, point.x, point.y)
    setLastInput(`glass → wipe at (${Math.round(point.x)}, ${Math.round(point.y)})`)
  }

  function onTouchMove(event: any) {
    const point = touchPoint(event)
    const current = gesture.current
    if (!point || !current) return
    if (current.onShelf) {
      if (Math.abs(point.x - current.startX) > TAP_SLOP || Math.abs(point.y - current.startY) > TAP_SLOP) {
        current.moved = true
      }
      return
    }
    const previous = stroke.current ?? point
    pendingSegments.current.push(previous.x, previous.y, point.x, point.y)
    stroke.current = point
  }

  function onTouchEnd(event: any) {
    const current = gesture.current
    gesture.current = null
    stroke.current = null
    if (!current || !current.onShelf || current.moved) return
    const point = touchPoint(event) ?? { x: current.startX, y: current.startY }
    const index = bottleAt(point.x, point.y, canvasCss.current.width)
    if (index >= 0) togglePick(index)
    else setLastInput(`shelf → nothing at (${Math.round(point.x)}, ${Math.round(point.y)}) (through clear glass)`)
  }

  function onTouchCancel() {
    gesture.current = null
    stroke.current = null
  }

  useEffect(() => {
    let stop: (() => void) | null = null
    let disposed = false
    let device: any = null

    async function boot() {
      const adapter = await gpu.requestAdapter()
      if (!adapter) throw new Error('no WebGPU adapter')
      device = await adapter.requestDevice()
      // The first error is the cause; the ones after it are usually consequences ("invalid due to a
      // previous error"). Keep the first and count the rest, so a strict runtime's verdict stays readable.
      let errorCount = 0
      device.onError((_error: any, text: string) => {
        errorCount += 1
        setStatus((current: string) => (errorCount === 1 ? text : `${current.split('\n(+')[0]}\n(+${errorCount - 1} more)`))
      })

      const context = gpu.getCanvasContext('main')
      const format = gpu.getPreferredCanvasFormat()
      context.configure({ device, format })

      const sampler = device.createSampler({
        magFilter: 'linear',
        minFilter: 'linear',
        addressModeU: 'clamp-to-edge',
        addressModeV: 'clamp-to-edge',
      })

      // ── The glyph atlas — the one thing an SDF cannot draw. Decoded natively, uploaded the spec way.
      const atlasBitmap = await createImageBitmap(await loadAsset('fog-labels.png'), { premultiplyAlpha: 'premultiply' })
      const atlasTexture = device.createTexture({
        label: 'fog-labels',
        size: { width: atlasBitmap.width, height: atlasBitmap.height },
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
      })
      device.queue.copyExternalImageToTexture({ source: atlasBitmap }, { texture: atlasTexture }, [atlasBitmap.width, atlasBitmap.height])
      atlasBitmap.close()

      // ── Pipelines
      const shelfModule = device.createShaderModule({ code: SHELF_SHADER, label: 'fog-shelf' })
      const shelfPipeline = device.createRenderPipeline({
        layout: 'auto',
        vertex: { module: shelfModule, entryPoint: 'vs_main' },
        fragment: { module: shelfModule, entryPoint: 'fs_main', targets: [{ format: 'rgba8unorm' }] },
      })
      const blurModule = device.createShaderModule({ code: BLUR_SHADER, label: 'fog-blur' })
      const blurPipeline = device.createRenderPipeline({
        layout: 'auto',
        vertex: { module: blurModule, entryPoint: 'vs_main' },
        fragment: { module: blurModule, entryPoint: 'fs_main', targets: [{ format: 'rgba8unorm' }] },
      })
      // The wipe field is r32float and read with textureLoad. An auto layout would infer a filterable
      // `float` sample type from `texture_2d<f32>` and a spec-conformant runtime (Dawn) then rejects the
      // r32float view — so the layout says unfilterable-float explicitly.
      const glassBindLayout = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: {} },
          { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
          { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
          { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
          { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
        ],
      })
      const glassModule = device.createShaderModule({ code: GLASS_SHADER, label: 'fog-glass' })
      const glassPipeline = device.createRenderPipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [glassBindLayout] }),
        vertex: { module: glassModule, entryPoint: 'vs_main' },
        fragment: { module: glassModule, entryPoint: 'fs_main', targets: [{ format }] },
      })

      // ── Uniforms
      const glassUniformBuffer = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
      const glassUniforms = new Float32Array(8)
      const shopUniformBuffer = device.createBuffer({ size: 16 + 320 + 320 + 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
      const shopUniforms = new Float32Array(184)
      ALL_BOTTLES.forEach((bottle, index) => {
        shopUniforms.set([...hexToRgb(bottle.color), 1], 4 + index * 4)
        shopUniforms.set([...hexToRgb(bottle.cap), 1], 84 + index * 4)
      })

      function writeShop(r: Resources) {
        shopUniforms[0] = r.pixelWidth
        shopUniforms[1] = r.pixelHeight
        shopUniforms[2] = r.pixelWidth / r.cssWidth
        shopUniforms[3] = r.cssWidth
        for (let index = 0; index < 20; index++) shopUniforms[164 + index] = picksRef.current.includes(index) ? 1 : 0
        device.queue.writeBuffer(shopUniformBuffer, 0, shopUniforms)
      }

      let resources: Resources | null = null

      function destroyResources(old: Resources) {
        old.wipeTexture.destroy()
        old.shelfTexture.destroy()
        old.blurA.destroy()
        old.blurB.destroy()
        for (const buffer of old.blurBuffers) buffer.destroy()
      }

      /** Everything that depends on the canvas size — rebuilt when it changes (rotation). */
      function buildResources(cssWidth: number, cssHeight: number, pixelWidth: number, pixelHeight: number): Resources {
        const field = new WipeField(cssWidth, cssHeight)
        const wipeTexture = device.createTexture({
          label: 'wipe-field',
          size: { width: field.columns, height: field.rows },
          format: 'r32float',
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        })
        // The shelf at full pixel size — it is what shows through clear glass, so it must be crisp.
        const shelfTexture = device.createTexture({
          label: 'shelf',
          size: { width: pixelWidth, height: pixelHeight },
          format: 'rgba8unorm',
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
        })
        // The blur at half CSS size — every pass keeps its taps one texel apart (a wider step skips
        // texels and combs the image into bands); reach comes from the small target and from stacking.
        const blurWidth = Math.max(1, Math.round(cssWidth / 2))
        const blurHeight = Math.max(1, Math.round(cssHeight / 2))
        const makeBlurTarget = (label: string) => device.createTexture({
          label,
          size: { width: blurWidth, height: blurHeight },
          format: 'rgba8unorm',
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
        })
        const blurA = makeBlurTarget('blur-a')
        const blurB = makeBlurTarget('blur-b')
        const passes: [any, any, number, number][] = [
          [shelfTexture, blurA, 1, 0],
          [blurA, blurB, 0, 1],
          [blurB, blurA, 1, 0],
          [blurA, blurB, 0, 1],
          [blurB, blurA, 1, 0],
          [blurA, blurB, 0, 1],
        ]
        const blurBuffers: any[] = []
        const blurPasses = passes.map(([source, target, dirX, dirY]) => {
          const buffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
          const sourceWidth = source === shelfTexture ? pixelWidth : blurWidth
          const sourceHeight = source === shelfTexture ? pixelHeight : blurHeight
          device.queue.writeBuffer(buffer, 0, new Float32Array([dirX, dirY, 1 / sourceWidth, 1 / sourceHeight]))
          blurBuffers.push(buffer)
          return {
            target,
            bindGroup: device.createBindGroup({
              layout: blurPipeline.getBindGroupLayout(0),
              entries: [
                { binding: 0, resource: { buffer } },
                { binding: 1, resource: source.createView() },
                { binding: 2, resource: sampler },
              ],
            }),
          }
        })

        const shelfBindGroup = device.createBindGroup({
          layout: shelfPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: shopUniformBuffer } },
            { binding: 1, resource: atlasTexture.createView() },
            { binding: 2, resource: sampler },
          ],
        })
        const glassBindGroup = device.createBindGroup({
          layout: glassBindLayout,
          entries: [
            { binding: 0, resource: { buffer: glassUniformBuffer } },
            { binding: 1, resource: shelfTexture.createView() },
            { binding: 2, resource: blurB.createView() },
            { binding: 3, resource: sampler },
            { binding: 4, resource: wipeTexture.createView() },
          ],
        })

        return {
          cssWidth, cssHeight, pixelWidth, pixelHeight, field, wipeTexture, shelfTexture,
          blurA, blurB, blurBuffers, blurPasses, shelfBindGroup, glassBindGroup,
          shelfDirty: true,
        }
      }

      function applyPreset(r: Resources, now: number, hold: number) {
        for (const path of PRESET_STROKES) {
          for (let i = 1; i < path.length; i++) {
            const [ax, ay] = path[i - 1]
            const [bx, by] = path[i]
            r.field.stamp(ax * r.cssWidth, ay * r.cssHeight, bx * r.cssWidth, by * r.cssHeight, now, hold, GROW_SECONDS)
          }
        }
      }

      let time = 0
      let size = context.getSize()
      let sizeCheck = 0
      let frames = 0
      let accumulated = 0
      let statsTick = STATS_INTERVAL
      const hold = preset.current ? PRESET_HOLD_SECONDS : HOLD_SECONDS

      stop = startFrameLoop(({ delta }: { delta: number }) => {
        if (disposed) return
        if (++sizeCheck >= 30) {
          sizeCheck = 0
          size = context.getSize()
        }
        if (size.width === 0 || size.height === 0) return
        const css = canvasCss.current
        if (css.width === 0 || css.height === 0) return

        if (!resources || resources.cssWidth !== css.width || resources.cssHeight !== css.height
          || resources.pixelWidth !== size.width || resources.pixelHeight !== size.height) {
          if (resources) destroyResources(resources)
          resources = buildResources(css.width, css.height, size.width, size.height)
          // The pane starts clear and fogs up like breath on glass — the field reads as "wiped just
          // now", so the regrowth curve does the fogging. The screenshot preset skips that.
          if (preset.current) applyPreset(resources, time, hold)
          else resources.field.fill(time - hold)
          const field = resources.field
          routing.current = {
            isClear: (x: number, y: number) => field.regrowAt(x, y, time, hold, GROW_SECONDS) < CLEAR_THRESHOLD,
          }
          statsTick = STATS_INTERVAL
        }
        const r = resources
        const step = Math.min(delta, 50) / 1000
        time += step

        // ── The finger
        const segments = pendingSegments.current
        for (let i = 0; i + 3 < segments.length; i += 4) {
          r.field.stamp(segments[i], segments[i + 1], segments[i + 2], segments[i + 3], time, hold, GROW_SECONDS)
        }
        segments.length = 0
        if (commands.current.wipeAll) {
          commands.current.wipeAll = false
          r.field.fill(time)
        }
        if (commands.current.fogAll) {
          commands.current.fogAll = false
          r.field.fill(time - hold)   // fogs back up gradually, the way it came
        }
        if (r.field.dirty) {
          r.field.dirty = false
          device.queue.writeTexture(
            { texture: r.wipeTexture }, r.field.values,
            { bytesPerRow: r.field.columns * 4 }, { width: r.field.columns, height: r.field.rows }
          )
          statsTick = STATS_INTERVAL
        }
        if (++statsTick >= STATS_INTERVAL) {
          statsTick = 0
          const percent = Math.round(r.field.clearFraction(time, hold, GROW_SECONDS) * 100)
          setClearPercent((current: number) => (current === percent ? current : percent))
        }

        // ── The shelf — redrawn when a pick changes, then blurred again
        const encoder = device.createCommandEncoder()
        if (r.shelfDirty || commands.current.shelfDirty) {
          r.shelfDirty = false
          commands.current.shelfDirty = false
          writeShop(r)
          const shelfPass = encoder.beginRenderPass({
            colorAttachments: [{ view: r.shelfTexture.createView(), loadOp: 'clear', storeOp: 'store' }],
          })
          shelfPass.setPipeline(shelfPipeline)
          shelfPass.setBindGroup(0, r.shelfBindGroup)
          shelfPass.draw(3)
          shelfPass.end()
          for (const blur of r.blurPasses) {
            const pass = encoder.beginRenderPass({
              colorAttachments: [{ view: blur.target.createView(), loadOp: 'clear', storeOp: 'store' }],
            })
            pass.setPipeline(blurPipeline)
            pass.setBindGroup(0, blur.bindGroup)
            pass.draw(3)
            pass.end()
          }
        }

        // ── The glass
        glassUniforms[0] = size.width
        glassUniforms[1] = size.height
        glassUniforms[2] = r.field.columns
        glassUniforms[3] = r.field.rows
        glassUniforms[4] = time
        glassUniforms[5] = hold
        glassUniforms[6] = GROW_SECONDS
        glassUniforms[7] = RESIDUE
        device.queue.writeBuffer(glassUniformBuffer, 0, glassUniforms)
        const pass = encoder.beginRenderPass({
          colorAttachments: [{
            view: context.getCurrentTexture().createView(),
            loadOp: 'clear',
            storeOp: 'store',
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
          }],
        })
        pass.setPipeline(glassPipeline)
        pass.setBindGroup(0, r.glassBindGroup)
        pass.draw(3)
        pass.end()
        device.queue.submit([encoder.finish()])

        frames += 1
        accumulated += delta
        if (accumulated >= 1000) {
          setFps(Math.round((frames * 1000) / accumulated))
          setClock(Math.round(time))
          frames = 0
          accumulated = 0
        }
      })
    }

    boot().catch((error) => setStatus(String(error?.message ?? error)))
    return () => {
      disposed = true
      if (stop) stop()
      if (device) device.destroy()
    }
  }, [])

  const pickedNames = picks.map((index: number) => ALL_BOTTLES[index].name)
  const total = picks.reduce((sum: number, index: number) => sum + Number(ALL_BOTTLES[index].price.replace(',', '')), 0)

  return (
    <view className="page">
      <webgpu-canvas
        canvas-id="main"
        className="canvas"
        pixel-ratio={2}
        bindcanvasresize={(event) => {
          const { width, height, pixelRatio } = event.detail
          canvasCss.current = { width: width / pixelRatio, height: height / pixelRatio }
        }}
        bindtouchstart={onTouchStart}
        bindtouchmove={onTouchMove}
        bindtouchend={onTouchEnd}
        bindtouchcancel={onTouchCancel}
      />

      <view className="hud">
        <text className="title">Condensation</text>
        <text className="subtitle" ios-platform-accessibility-id="fog-clock">wipe it · taps reach the shelf where it is clear · {fps} fps · {clock}s</text>
        <text className="note" ios-platform-accessibility-id="fog-input">{`last input: ${lastInput}`}</text>
        <text className="note" ios-platform-accessibility-id="fog-state">
          {`clear glass ${clearPercent}% · picked ${picks.length}${picks.length ? ` · ${formatWon(total)} · last ${pickedNames[pickedNames.length - 1]}` : ''}`}
        </text>
        {status ? <text className="status">{status}</text> : null}
      </view>
      <text className="badge">WebGPU on Lynx</text>

      <view className="controls">
        <view
          className="control-button"
          ios-platform-accessibility-id="fog-button-fog"
          bindtap={() => { commands.current.fogAll = true }}
        >
          <text ios-platform-accessibility-id="fog-button-fog-label">fog it all</text>
        </view>
        <view
          className="control-button"
          ios-platform-accessibility-id="fog-button-wipe"
          bindtap={() => { commands.current.wipeAll = true }}
        >
          <text ios-platform-accessibility-id="fog-button-wipe-label">wipe it all</text>
        </view>
      </view>
    </view>
  )
}

root.render(<FogScene />)
