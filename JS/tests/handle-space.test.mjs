/**
 * 핸들 공간 — **모듈 전체가 하나를 쓴다.**
 *
 * 네이티브 레지스트리는 컨텍스트당 하나이고 핸들 정수만으로 객체를 찾는다. 카운터를
 * 디바이스마다 두면 두 번째 디바이스가 1번부터 다시 발급해 **첫 디바이스의 객체를 조용히
 * 덮어쓴다** — 오류가 나지 않고 남의 버퍼에 그리게 되는, 화면에서만 드러나는 종류의 버그다.
 *
 * 다중 디바이스는 흔한 패턴이 아니지만 우리 코드에도 이미 있다 (`spec` 데모 씬이
 * `requiredFeatures`를 검증하려고 두 번째 디바이스를 만든다).
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { installNativeMock, makeDevice } from './helpers.mjs';
import gpu, { createImageBitmap } from '../webgpu.js';

test('디바이스가 둘이어도 핸들이 겹치지 않는다', async () => {
  installNativeMock();
  const adapter = await gpu.requestAdapter();
  const deviceA = await adapter.requestDevice();
  const deviceB = await adapter.requestDevice();

  const first = deviceA.createBuffer({ size: 16, usage: 0x0004 });
  const second = deviceB.createBuffer({ size: 16, usage: 0x0004 });
  const third = deviceA.createBuffer({ size: 16, usage: 0x0004 });

  const handles = new Set([first.id, second.id, third.id]);
  assert.equal(handles.size, 3, `핸들이 겹쳤다: ${first.id}, ${second.id}, ${third.id}`);
});

test('한 디바이스 안의 여러 종류도 서로 겹치지 않는다', async () => {
  installNativeMock();
  const device = await makeDevice();

  const handles = [
    device.createBuffer({ size: 16, usage: 0x0004 }).id,
    device.createTexture({ size: [4, 4], format: 'rgba8unorm', usage: 0x04 }).id,
    device.createSampler().id,
    device.createShaderModule({ code: '@fragment fn fs() {}' }).id,
  ];
  assert.equal(new Set(handles).size, handles.length, `겹친 핸들: ${handles.join(', ')}`);
});

test('createImageBitmap의 핸들은 활성 디바이스와 무관하게 유일하다', async () => {
  // 여기가 원래 지적된 자리다 — 전역 `createImageBitmap`이 "마지막에 만든 디바이스"의
  // 카운터를 쓰면, 첫 디바이스의 객체 번호와 겹칠 수 있었다.
  installNativeMock({ decodeImageResult: { ok: true, width: 4, height: 4 } });
  const adapter = await gpu.requestAdapter();
  const deviceA = await adapter.requestDevice();
  const early = deviceA.createBuffer({ size: 16, usage: 0x0004 });

  await adapter.requestDevice();   // 두 번째 디바이스가 활성이 된다

  const bitmap = await createImageBitmap(new ArrayBuffer(8));
  const later = deviceA.createBuffer({ size: 16, usage: 0x0004 });

  assert.notEqual(bitmap.id, early.id, '이미지가 첫 디바이스의 버퍼를 덮어쓴다');
  assert.notEqual(bitmap.id, later.id);
  assert.notEqual(early.id, later.id);
});

test('destroy 뒤에도 번호를 재사용하지 않는다', async () => {
  // 재사용하면, 파괴된 객체의 번호를 아직 들고 있는 JS 객체가 **남의 자리**를 가리킨다.
  installNativeMock();
  const device = await makeDevice();

  const first = device.createBuffer({ size: 16, usage: 0x0004 });
  first.destroy();
  const second = device.createBuffer({ size: 16, usage: 0x0004 });

  assert.notEqual(second.id, first.id);
});
