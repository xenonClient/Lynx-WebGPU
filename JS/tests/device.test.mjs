/**
 * `GPUDevice`의 명세 표면 — `features` / `lost`.
 *
 * Three.js `WebGPUBackend.init()`이 어댑터에서 고른 기능을 `requiredFeatures`로 요청한 뒤
 * `device.features`를 다시 읽고, 곧바로 `device.lost.then(...)`을 건다. 이 두 속성이 없으면
 * 렌더러 부트스트랩이 TypeError로 죽으므로, 미지원 결정과 별개로 속성 자체를 보장한다.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import gpu from '../webgpu.js';
import { installNativeMock, makeDevice } from './helpers.mjs';

/** 어댑터가 기능 목록을 보고하는 목. */
function mockWithFeatures(features) {
  const state = installNativeMock();
  const base = globalThis.NativeModules.WebGPU.adapterInfo;
  globalThis.NativeModules.WebGPU.adapterInfo = () => ({ ...base(), features });
  return state;
}

test('요청한 기능만 device.features에 들어간다', async () => {
  mockWithFeatures(['timestamp-query', 'shader-f16']);
  const adapter = await gpu.requestAdapter();

  const device = await adapter.requestDevice({ requiredFeatures: ['timestamp-query'] });

  assert.equal(device.features.has('timestamp-query'), true);
  assert.equal(device.features.has('shader-f16'), false, '요청하지 않은 기능은 빠져야 한다');
  assert.equal(device.features.size, 1);
});

test('요청이 없으면 device.features는 비어 있되 존재한다', async () => {
  installNativeMock();
  const device = await makeDevice();

  assert.equal(device.features.has('timestamp-query'), false);
  assert.equal(device.features.size, 0);
});

test('어댑터가 지원하지 않는 기능을 요구하면 거부한다', async () => {
  mockWithFeatures(['timestamp-query']);
  const adapter = await gpu.requestAdapter();

  await assert.rejects(
    adapter.requestDevice({ requiredFeatures: ['shader-f16'] }),
    /shader-f16/,
    '무엇이 없는지 이름을 알려 줘야 한다'
  );
});

test('adapter.info와 device.adapterInfo가 명세 모양이다', async () => {
  const state = installNativeMock();
  const base = globalThis.NativeModules.WebGPU.adapterInfo;
  globalThis.NativeModules.WebGPU.adapterInfo = () => ({
    ...base(),
    info: { vendor: 'apple', architecture: 'apple-8', description: 'Apple M3' },
  });

  const adapter = await gpu.requestAdapter();
  const device = await adapter.requestDevice();

  assert.equal(adapter.info.vendor, 'apple');
  assert.equal(adapter.info.architecture, 'apple-8');
  assert.equal(adapter.info.description, 'Apple M3');
  // 모르는 자리는 빈 문자열이다 — 지어내면 그 값으로 분기하는 코드가 잘못된 우회로를 탄다.
  assert.equal(adapter.info.device, '');
  assert.equal(adapter.info.isFallbackAdapter, false);
  assert.equal(adapter.info.subgroupMinSize, 0);
  // 명세는 device에도 같은 것을 노출한다.
  assert.equal(device.adapterInfo, adapter.info);
  assert.ok(state);
});

test('네이티브가 info를 안 주면 전부 빈 값이다 (undefined가 새지 않는다)', async () => {
  installNativeMock();
  const adapter = await gpu.requestAdapter();

  assert.equal(adapter.info.vendor, '');
  assert.equal(adapter.info.architecture, '');
  assert.equal(adapter.info.isFallbackAdapter, false);
});

test('device.lost는 then을 걸 수 있는 Promise다', async () => {
  installNativeMock();
  const device = await makeDevice();

  assert.ok(device.lost instanceof Promise, 'lost가 Promise여야 한다');
  // 영원히 pending이어야 한다 — 유실을 보고하지 않는 구현이 resolve하면
  // 웹 코드의 유실 핸들러(재초기화 등)가 멀쩡한 디바이스에서 돈다.
  const settled = await Promise.race([
    device.lost.then(() => 'settled'),
    new Promise((resolve) => setTimeout(() => resolve('pending'), 20)),
  ]);
  assert.equal(settled, 'pending');
});
