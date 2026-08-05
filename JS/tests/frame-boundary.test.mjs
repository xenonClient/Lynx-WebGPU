/**
 * 프레임 경계 — present는 `submit()`이 아니라 **프레임 루프 콜백의 끝**에서 일어난다.
 *
 * 브라우저가 그렇게 한다: 한 프레임 안에서 submit을 몇 번 하든 캔버스는 태스크가 끝난 뒤에
 * 한 번 나간다. 그래서 웹 라이브러리는 한 프레임에 여러 번 submit하면서 드로어블 텍스처
 * 뷰를 그 프레임 내내 재사용한다 (three.js `PostProcessing`: 씬 패스 → bloom 밉 체인 →
 * 출력 패스).
 *
 * submit마다 present하면 첫 submit이 드로어블을 내보내고 뷰를 만료시켜, 같은 프레임의 남은
 * 패스가 "없는 핸들"로 통째로 거부된다 — `threelab` 데모 씬이 그걸로 깨졌다.
 *
 * 틱은 기기와 같은 경로(`GlobalEventEmitter`)로 **동기로** 몬다. 타이머 폴백을 쓰면 콜백이
 * 던졌을 때 그 오류가 타이머 밖으로 새어 테스트 러너가 먼저 가져간다.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { installNativeMock, makeDevice, installFrameEmitter } from './helpers.mjs';
import gpu, { startFrameLoop } from '../webgpu.js';

/** 프레임 이미터를 깔고 본문을 돌린 뒤 반드시 걷어낸다. */
function withFrames(body) {
  return async () => {
    const emitter = installFrameEmitter();
    try {
      await body(emitter);
    } finally {
      emitter.uninstall();
    }
  };
}

/** 핸들러를 걸고 틱 하나를 몬 뒤 정리한다. */
function oneTick(emitter, handler) {
  const stop = startFrameLoop(handler, { fps: 240 });
  try {
    emitter.tick();
  } finally {
    stop();
  }
}

/** 프레임 안에서 GPU 작업 한 조각. */
function doSomeWork(device) {
  device.queue.writeBuffer(device.createBuffer({ size: 4, usage: 0x0008 }), 0, new Uint8Array(4));
  device.queue.submit([]);
}

test('틱 안의 submit은 present를 미룬다', withFrames(async (emitter) => {
  const state = installNativeMock();
  const device = await makeDevice();

  oneTick(emitter, () => {
    doSomeWork(device);
    doSomeWork(device);
  });

  const presents = state.executeCalls.filter((call) => call.present !== false);
  assert.equal(presents.length, 1, `틱당 present는 한 번이어야 한다 (${presents.length}번 나갔다)`);
  // 앞선 두 배치는 명령을 싣고 나가되 present는 하지 않는다 — GPU 작업을 미루지는 않는다.
  const deferred = state.executeCalls.filter((call) => call.present === false);
  assert.equal(deferred.length, 2);
  assert.ok(deferred.every((call) => call.commands.length > 0), '명령은 그때그때 나가야 한다');
}));

test('마무리 present는 명령이 없어도 나간다', withFrames(async (emitter) => {
  // 드로어블을 내보내는 것이 목적이라 마지막 배치는 비어 있을 수 있다.
  const state = installNativeMock();
  const device = await makeDevice();

  oneTick(emitter, () => doSomeWork(device));

  const last = state.executeCalls[state.executeCalls.length - 1];
  assert.equal(last.present, true);
  assert.equal(last.commands.length, 0, '명령은 앞 배치가 이미 가져갔다');
}));

test('GPU 작업이 없던 틱은 브리지를 건너지 않는다', withFrames(async (emitter) => {
  const state = installNativeMock();
  await makeDevice();

  oneTick(emitter, () => {});

  assert.equal(state.executeCalls.length, 0);
}));

test('틱 밖의 submit은 그 자리에서 present한다', withFrames(async () => {
  // 프레임 루프 없이 직접 그리는 코드(테스트 하네스·오프스크린)는 예전 그대로여야 한다.
  const state = installNativeMock();
  const device = await makeDevice();

  doSomeWork(device);

  assert.equal(state.executeCalls.length, 1);
  assert.notEqual(state.executeCalls[0].present, false);
}));

test('내부 제출(popErrorScope)은 틱 안에서도 present=false 그대로다', withFrames(async (emitter) => {
  const state = installNativeMock();
  const device = await makeDevice();

  oneTick(emitter, () => {
    device.pushErrorScope('validation');
    device.popErrorScope();
  });

  const internal = state.executeCalls.filter((call) => call.present === false);
  assert.ok(internal.length > 0, '내부 제출이 없다');
}));

test('콜백이 던져도 present는 나간다 — 화면이 그 프레임에서 멈추면 안 된다', withFrames(async (emitter) => {
  const state = installNativeMock();
  const device = await makeDevice();

  const stop = startFrameLoop(() => {
    doSomeWork(device);
    throw new Error('프레임 안에서 터졌다');
  }, { fps: 240 });

  // 오류는 **삼키지 않는다** — present만 내보내고 그대로 위로 던진다.
  assert.throws(() => emitter.tick(), /프레임 안에서 터졌다/);
  stop();

  const presents = state.executeCalls.filter((call) => call.present !== false);
  assert.equal(presents.length, 1, '오류가 나도 그 프레임은 화면에 나가야 한다');
}));

test('던진 틱 다음에도 프레임 경계가 살아 있다', withFrames(async (emitter) => {
  // 깊이 카운터가 새면 이후 모든 프레임이 present를 못 받아 화면이 멈춘다.
  const state = installNativeMock();
  const device = await makeDevice();

  let shouldThrow = true;
  const stop = startFrameLoop(() => {
    doSomeWork(device);
    if (shouldThrow) {
      shouldThrow = false;
      throw new Error('한 번만 터진다');
    }
  }, { fps: 240 });

  assert.throws(() => emitter.tick());
  emitter.tick();
  stop();

  const presents = state.executeCalls.filter((call) => call.present !== false);
  assert.equal(presents.length, 2, '틱마다 한 번씩');
}));

test('한 구독자가 멈춰도 다른 루프는 계속 돈다', withFrames(async (emitter) => {
  // 네이티브 티커는 하나뿐이다. 세지 않으면 잠깐 돌린 루프가 `stop()`될 때
  // rAF 펌프(`installAnimationFrame`)까지 같이 죽어 **화면이 조용히 멈춘다**.
  const state = installNativeMock();
  const device = await makeDevice();

  let longLived = 0;
  const stopLong = startFrameLoop(() => { longLived += 1; doSomeWork(device); });
  const stopShort = startFrameLoop(() => {});

  emitter.tick();
  stopShort();          // 짧게 쓰고 끄는 쪽
  emitter.tick();
  emitter.tick();
  stopLong();

  assert.equal(longLived, 3, '남의 stop()에 끌려 멈췄다');
  assert.equal(state.stopFrameLoopCalls, 1, '마지막 구독자가 놓을 때만 네이티브를 멈춘다');
}));

test('stop()을 두 번 불러도 남의 구독을 깎지 않는다', withFrames(async (emitter) => {
  const state = installNativeMock();
  const device = await makeDevice();

  const stopA = startFrameLoop(() => doSomeWork(device));
  const stopB = startFrameLoop(() => {});
  stopB();
  stopB();              // 두 번째는 아무 일도 없어야 한다

  emitter.tick();
  assert.equal(state.stopFrameLoopCalls, 0, '아직 A가 남아 있다');
  stopA();
  assert.equal(state.stopFrameLoopCalls, 1);
}));

test('다음 틱은 빚을 새로 센다', withFrames(async (emitter) => {
  const state = installNativeMock();
  const device = await makeDevice();

  const stop = startFrameLoop(() => doSomeWork(device), { fps: 240 });
  emitter.tick();
  emitter.tick();
  stop();

  const presents = state.executeCalls.filter((call) => call.present !== false);
  assert.equal(presents.length, 2);
}));

// ---------------------------------------------------------------------------
// 명세 "Expire the current texture" (W3C WebGPU §canvas rendering)
//
//   getCurrentTexture(): `[[currentTexture]]`가 있으면 **그대로 돌려준다.**
//   만료를 부르는 자리: presentation · configure() · 캔버스 리사이즈.
// ---------------------------------------------------------------------------

test('한 프레임 안에서 getCurrentTexture는 같은 텍스처를 준다', withFrames(async (emitter) => {
  // 호출마다 새 텍스처를 내면, 뷰를 캐시해 두는 웹 코드가 매 패스 다른 텍스처를 보게 된다.
  installNativeMock();
  const device = await makeDevice();
  const context = gpu.getCanvasContext('main');
  context.configure({ device, format: 'bgra8unorm' });

  let first;
  let second;
  oneTick(emitter, () => {
    first = context.getCurrentTexture();
    second = context.getCurrentTexture();
    device.queue.submit([]);
  });

  assert.equal(first, second, '같은 프레임인데 다른 텍스처가 나왔다');
  assert.equal(first.id, second.id);
}));

test('present하면 만료되어 다음 프레임은 새 텍스처다', withFrames(async (emitter) => {
  installNativeMock();
  const device = await makeDevice();
  const context = gpu.getCanvasContext('main');
  context.configure({ device, format: 'bgra8unorm' });

  const seen = [];
  const stop = startFrameLoop(() => {
    seen.push(context.getCurrentTexture().id);
    device.queue.submit([]);
  });
  emitter.tick();
  emitter.tick();
  stop();

  assert.equal(seen.length, 2);
  assert.notEqual(seen[0], seen[1], 'present 뒤에도 옛 텍스처를 주면 화면이 멈춘다');
}));

test('configure()도 현재 텍스처를 만료시킨다', withFrames(async () => {
  installNativeMock();
  const device = await makeDevice();
  const context = gpu.getCanvasContext('main');
  context.configure({ device, format: 'bgra8unorm' });

  const before = context.getCurrentTexture();
  context.configure({ device, format: 'rgba8unorm' });
  const after = context.getCurrentTexture();

  assert.notEqual(before.id, after.id);
  assert.equal(after.format, 'rgba8unorm', '새 설정으로 받아야 한다');
}));

test('설정 전 getCurrentTexture는 InvalidStateError다', withFrames(async () => {
  installNativeMock();
  await makeDevice();
  const context = gpu.getCanvasContext('unconfigured-probe');

  assert.throws(() => context.getCurrentTexture(), (error) => {
    assert.equal(error.name, 'InvalidStateError', '명세가 정한 이름이다');
    return true;
  });
}));

test('캔버스 크기가 바뀌면 만료된다', withFrames(async (emitter) => {
  // 크기가 달라진 드로어블을 옛 텍스처로 계속 가리키면 다음 패스가 어긋난 크기로 그린다.
  let size = { width: 300, height: 150 };
  const state = installNativeMock();
  state.executeResult = () => ({ ok: true, canvases: { main: size } });
  const device = await makeDevice();
  const context = gpu.getCanvasContext('main');
  context.configure({ device, format: 'bgra8unorm' });

  let first;
  let second;
  oneTick(emitter, () => {
    first = context.getCurrentTexture();
    size = { width: 400, height: 200 };   // 다음 제출 응답에서 크기가 바뀐다
    device.queue.submit([]);
    second = context.getCurrentTexture();
  });

  assert.notEqual(first.id, second.id, '리사이즈 뒤에도 옛 텍스처를 줬다');
}));
