/**
 * `loadAsset`'s contract — the name goes to native as is, the received ArrayBuffer comes back undecoded,
 * and a failure rejects with the first error message.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';

import { loadAsset } from '../webgpu.js';
import { installNativeMock, randomBytes } from './helpers.mjs';

test('passes the name through and returns native\'s ArrayBuffer undecoded', async () => {
  const bytes = randomBytes(64);
  const state = installNativeMock({
    loadAssetResult: { ok: true, data: bytes.buffer, byteLength: bytes.byteLength },
  });

  const buffer = await loadAsset('LUTs/neutral.cube');

  assert.deepEqual(state.loadAssetCalls, [{ name: 'LUTs/neutral.cube' }]);
  assert.equal(buffer, bytes.buffer); // the same object, with no copy or conversion
});

test('an ok:false from native rejects with the first error message', async () => {
  installNativeMock({
    loadAssetResult: {
      ok: false,
      errors: [{ kind: 'validation', message: "'absent.bin' is not in the bundle" }],
    },
  });

  await assert.rejects(loadAsset('absent.bin'), /absent\.bin/);
});

test('no response at all (undefined) rejects with the name included', async () => {
  installNativeMock({ loadAssetResult: () => undefined });

  await assert.rejects(loadAsset('ghost.bin'), /ghost\.bin/);
});
