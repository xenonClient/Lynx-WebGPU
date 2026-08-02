/**
 * `loadAsset`의 계약 — 이름을 그대로 네이티브에 넘기고, 받은 ArrayBuffer를
 * 디코딩 없이 돌려주며, 실패는 첫 오류 메시지로 reject된다.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';

import { loadAsset } from '../webgpu.js';
import { installNativeMock, randomBytes } from './helpers.mjs';

test('이름을 그대로 넘기고 네이티브의 ArrayBuffer를 디코딩 없이 돌려준다', async () => {
  const bytes = randomBytes(64);
  const state = installNativeMock({
    loadAssetResult: { ok: true, data: bytes.buffer, byteLength: bytes.byteLength },
  });

  const buffer = await loadAsset('LUTs/neutral.cube');

  assert.deepEqual(state.loadAssetCalls, [{ name: 'LUTs/neutral.cube' }]);
  assert.equal(buffer, bytes.buffer); // 복사·변환 없이 그 객체 그대로
});

test('네이티브가 ok:false를 주면 첫 오류 메시지로 reject된다', async () => {
  installNativeMock({
    loadAssetResult: {
      ok: false,
      errors: [{ kind: 'validation', message: "번들에 'absent.bin'이(가) 없다" }],
    },
  });

  await assert.rejects(loadAsset('absent.bin'), /absent\.bin/);
});

test('응답이 아예 없으면(undefined) 이름을 담아 reject된다', async () => {
  installNativeMock({ loadAssetResult: () => undefined });

  await assert.rejects(loadAsset('ghost.bin'), /ghost\.bin/);
});
