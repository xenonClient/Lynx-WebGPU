import { root } from '@lynx-js/react'
import { DemoScene, type SceneContext } from '../scene.jsx'
import { GPUBufferUsage, GPUMapMode, GPUTextureUsage } from '../webgpu.js'

/** 팔레트 색 수. 셰이더는 이 값을 모른다 — `arrayLength()`로 직접 센다. */
const PALETTE = 7
const FRAME_SIZE = 8

/**
 * 팔레트를 애니메이션한다.
 *
 * 개수를 유니폼으로 받지 않고 `arrayLength(&palette)`로 센다. 런타임이 예약 인덱스에 꽂아 주는
 * 버퍼 크기 표가 틀리면 여기서 바로 어긋난다 (`info[0]`으로 CPU가 확인한다).
 */
const COMPUTE_SHADER = /* wgsl */ `
struct Params {
  time: f32,
};
@group(0) @binding(0) var<storage, read_write> palette: array<vec4f>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read_write> info: array<u32>;

@compute @workgroup_size(8)
fn animate(@builtin(global_invocation_id) id: vec3u) {
  let count = arrayLength(&palette);
  if (id.x == 0u) {
    info[0] = count;
    info[1] = arrayLength(&info);
  }
  if (id.x >= count) {
    return;
  }

  let t = f32(id.x) / f32(count);
  let phase = params.time * 0.3 + t;
  palette[id.x] = vec4f(
    0.5 + 0.5 * sin(6.2831853 * (phase + 0.00)),
    0.5 + 0.5 * sin(6.2831853 * (phase + 0.33)),
    0.5 + 0.5 * sin(6.2831853 * (phase + 0.67)),
    1.0,
  );
}
`

/**
 * 화면을 세 띠로 나눠 세 기능을 한 번에 보여 준다.
 *
 * 위: 타입 없는 상수식(AbstractInt)과 함수 지역 `const`로 잡은 배열.
 * 가운데: 같은 텍스처를 왼쪽은 `textureSampleBaseClampToEdge`(외부 텍스처), 오른쪽은 보통 샘플링.
 *         repeat 샘플러에 좌표를 일부러 [0,1] 밖으로 넘겨서 **오른쪽만 반대쪽이 감겨 들어온다**.
 * 아래: `arrayLength()`가 정한 칸 수 — 길이가 틀리면 칸 수가 눈에 보이게 달라진다.
 */
const RENDER_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> palette: array<vec4f>;
@group(0) @binding(1) var frame: texture_external;
@group(0) @binding(2) var frameSampler: sampler;
@group(0) @binding(3) var frame2d: texture_2d<f32>;

// 타입을 적지 않은 상수식은 WGSL에서 추상 정수(AbstractInt)다 — 쓰이는 자리의 타입으로 굳는다.
const HIGHLIGHT = vec3(1);
const TILES = vec2(8, 2);

struct Out {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> Out {
  var corners = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var out: Out;
  out.position = vec4f(corners[index], 0.0, 1.0);
  out.uv = corners[index] * vec2f(0.5, -0.5) + vec2f(0.5, 0.5);
  return out;
}

// 함수 안에서 선언한 const도 컴파일 타임 상수 — 배열 크기로 쓸 수 있다.
fn ramp(t: f32) -> vec3f {
  const STEPS = 5u;
  var stops: array<vec3f, STEPS>;
  // HUD가 얹히는 위쪽은 어둡게, 아래로 갈수록 밝게 — 흰 글씨가 읽히도록.
  stops[0] = vec3f(0.06, 0.07, 0.13);
  stops[1] = vec3f(0.09, 0.16, 0.31);
  stops[2] = vec3f(0.12, 0.31, 0.36);
  stops[3] = vec3f(0.85, 0.55, 0.25);
  stops[4] = vec3f(0.95, 0.34, 0.40);
  let index = min(u32(t * f32(STEPS)), STEPS - 1u);
  return stops[index];
}

// uv * TILES — 추상 정수 벡터가 f32 문맥에서 f32로 굳는다.
fn checker(uv: vec2f) -> f32 {
  let cell = floor(uv * TILES);
  return select(0.82, 1.0, (u32(cell.x) + u32(cell.y)) % 2u == 0u);
}

@fragment
fn fs_main(in: Out) -> @location(0) vec4f {
  let uv = in.uv;

  // 위: 타입 없는 상수식. HUD가 얹히는 자리라 어두운 쪽이 위로 오게 두었다.
  if (uv.y < 0.40) {
    let t = uv.y / 0.40;
    let base = ramp(t) * checker(vec2f(uv.x, t));
    // HIGHLIGHT는 vec3(1) — 같은 상수가 f32 벡터 자리에서 f32로 굳는다.
    return vec4f(mix(base, HIGHLIGHT, 0.06), 1.0);
  }

  // 가운데: 같은 텍스처 · 같은 좌표를 왼쪽만 가장자리 클램프로 샘플한다.
  if (uv.y < 0.76) {
    let local = vec2f(fract(uv.x * 2.0), (uv.y - 0.40) / 0.36);
    // 일부러 [0,1] 밖으로 넘긴다 — repeat 샘플러라 클램프가 없으면 반대쪽이 감겨 들어온다.
    let coord = local * 1.30 - vec2f(0.15, 0.15);
    // 두 패널 경계에 가는 선을 둬서 좌우 비교임을 분명히 한다.
    if (abs(uv.x - 0.5) < 0.004) {
      return vec4f(0.04, 0.05, 0.08, 1.0);
    }
    if (uv.x < 0.5) {
      return textureSampleBaseClampToEdge(frame, frameSampler, coord);
    }
    return textureSample(frame2d, frameSampler, coord);
  }

  // 아래: arrayLength()가 정한 칸 수 — 길이가 틀리면 칸 수가 달라진다.
  let count = arrayLength(&palette);
  let index = min(u32(uv.x * f32(count)), count - 1u);
  let edge = fract(uv.x * f32(count));
  let seam = smoothstep(0.0, 0.04, edge) * smoothstep(1.0, 0.96, edge);
  return vec4f(palette[index].rgb * mix(0.5, 1.0, seam), 1.0);
}
`

/** 8x8 "비디오 프레임". 가장자리 열/행을 대비색으로 칠해 이음선이 생기면 바로 보이게 한다. */
function makeFrame(step: number): Uint8Array {
  const pixels = new Uint8Array(FRAME_SIZE * FRAME_SIZE * 4)
  const highlight = step % FRAME_SIZE
  for (let y = 0; y < FRAME_SIZE; y += 1) {
    for (let x = 0; x < FRAME_SIZE; x += 1) {
      const offset = (y * FRAME_SIZE + x) * 4
      let r = 40 + Math.round((x / (FRAME_SIZE - 1)) * 60)
      let g = 90 + Math.round((y / (FRAME_SIZE - 1)) * 90)
      let b = 150 - Math.round((x / (FRAME_SIZE - 1)) * 40)
      if (x === highlight) {
        r = 250
        g = 240
        b = 180
      }
      if (x === 0) {
        r = 255
        g = 40
        b = 200
      } // 왼쪽 끝: 자홍
      if (x === FRAME_SIZE - 1) {
        r = 30
        g = 220
        b = 120
      } // 오른쪽 끝: 초록
      if (y === 0) {
        g = Math.min(255, g + 60)
      }
      if (y === FRAME_SIZE - 1) {
        r = Math.round(r * 0.35)
        g = Math.round(g * 0.35)
        b = Math.round(b * 0.35)
      }
      pixels[offset] = r
      pixels[offset + 1] = g
      pixels[offset + 2] = b
      pixels[offset + 3] = 255
    }
  }
  return pixels
}

function setup({ device, context, format, report }: SceneContext) {
  const computeModule = device.createShaderModule({ code: COMPUTE_SHADER, label: 'wgsl.animate' })
  const renderModule = device.createShaderModule({ code: RENDER_SHADER, label: 'wgsl.render' })

  const palette = device.createBuffer({
    size: PALETTE * 16,
    usage: GPUBufferUsage.STORAGE,
    label: 'palette',
  })
  const params = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })
  const info = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    label: 'info',
  })
  // MAP_READ는 COPY_DST와만 조합할 수 있다(명세) — 리드백은 전용 스테이징 버퍼로 받는다.
  const infoStaging = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    label: 'info.staging',
  })

  // 외부 텍스처 자리에는 보통 텍스처 뷰를 그대로 묶는다 (한 면짜리 비디오 프레임과 같은 모양).
  const frame = device.createTexture({
    size: { width: FRAME_SIZE, height: FRAME_SIZE },
    format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    label: 'frame',
  })
  const frameView = frame.createView()
  // repeat 샘플러 — 클램프가 없으면 반대쪽 가장자리가 감겨 들어온다.
  const frameSampler = device.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
    addressModeU: 'repeat',
    addressModeV: 'repeat',
  })

  const computePipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module: computeModule, entryPoint: 'animate' },
  })
  const computeBindGroup = device.createBindGroup({
    layout: computePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: palette } },
      { binding: 1, resource: { buffer: params } },
      { binding: 2, resource: { buffer: info } },
    ],
  })

  const renderPipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module: renderModule, entryPoint: 'vs_main' },
    fragment: { module: renderModule, entryPoint: 'fs_main', targets: [{ format }] },
  })
  const renderBindGroup = device.createBindGroup({
    layout: renderPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: palette } },
      { binding: 1, resource: frameView },
      { binding: 2, resource: frameSampler },
      { binding: 3, resource: frameView },
    ],
  })

  const paramsData = new Float32Array(4)
  let time = 0
  let frames = 0
  let reading = false
  let reported = false

  return ({ delta }: { delta: number }) => {
    time += delta / 1000
    paramsData[0] = time
    device.queue.writeBuffer(params, 0, paramsData)

    // 프레임을 갱신해 "비디오"처럼 흐르게 한다 (8x8이라 256바이트).
    if (frames % 6 === 0) {
      device.queue.writeTexture(
        { texture: frame },
        makeFrame(Math.floor(frames / 6)),
        { bytesPerRow: FRAME_SIZE * 4 },
        { width: FRAME_SIZE, height: FRAME_SIZE }
      )
    }

    const encoder = device.createCommandEncoder()

    const compute = encoder.beginComputePass()
    compute.setPipeline(computePipeline)
    compute.setBindGroup(0, computeBindGroup)
    compute.dispatchWorkgroups(1)
    compute.end()

    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
          loadOp: 'clear',
          storeOp: 'store',
          clearValue: { r: 0.04, g: 0.05, b: 0.08, a: 1 },
        },
      ],
    })
    pass.setPipeline(renderPipeline)
    pass.setBindGroup(0, renderBindGroup)
    pass.draw(3)
    pass.end()

    // 셰이더가 센 길이를 CPU가 한 번 확인한다 — 크기 표가 맞는지 눈이 아니라 숫자로 본다.
    frames += 1
    const wantsReadback = !reported && frames > 10 && !reading
    if (wantsReadback) encoder.copyBufferToBuffer(info, 0, infoStaging, 0, 16)

    device.queue.submit([encoder.finish()])

    if (wantsReadback) {
      reading = true
      infoStaging
        .mapAsync(GPUMapMode.READ)
        .then((buffer: ArrayBuffer) => {
          const values = new Uint32Array(buffer)
          const ok = values[0] === PALETTE && values[1] === 4
          reported = ok
          report(
            `arrayLength: palette ${values[0]} (기대 ${PALETTE}) · info ${values[1]} (기대 4)` +
              (ok ? ' ✓' : ' ✗')
          )
        })
        .catch((error: unknown) => report(`리드백 실패: ${String(error)}`))
        .finally(() => {
          infoStaging.unmap()
          reading = false
        })
    }
  }
}

root.render(
  <DemoScene
    title="WGSL 호환성"
    subtitle="arrayLength · 외부 텍스처(왼쪽만 가장자리 클램프) · 타입 없는 상수식"
    setup={setup}
  />
)
