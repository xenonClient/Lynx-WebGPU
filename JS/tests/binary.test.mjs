/**
 * Binary path verification — whether the `data` put on a command is **a real `ArrayBuffer`**, and whether
 * the bytes inside it are exactly the requested range.
 *
 * The type is the crux. Lynx's value converter only recognizes an `ArrayBuffer` and treats a view
 * (TypedArray) as a plain object, turning it into `{"0":1,…}`, so a leaked view breaks **silently with no
 * error**. That is why `instanceof ArrayBuffer` is asserted alongside every time.
 *
 * The codec is an internal function, so it is checked through the public API:
 * the upload side is the `data` `queue.writeBuffer` puts on, the download side is `buffer.mapAsync`'s return value.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { GPUBufferUsage, GPUMapMode } from '../webgpu.js';
import { installNativeMock, makeDevice, commandsOf, randomBytes } from './helpers.mjs';

const LENGTHS = [0, 1, 2, 3, 4, 5, 255, 256, 3071, 3072, 3073, 100_000];

/** Pulls the `data` off a command, checking the type along the way. */
function payloadOf(state, op) {
  const command = commandsOf(state).find((entry) => entry.op === op);
  assert.ok(command, `there is no ${op} command`);
  assert.ok(
    command.data instanceof ArrayBuffer,
    `${op}'s data is not an ArrayBuffer (${Object.prototype.toString.call(command.data)}) — ` +
      'a TypedArray leaking through gets turned into an object by Lynx and breaks silently'
  );
  return new Uint8Array(command.data);
}

async function uploadedByShim(bytes) {
  const state = installNativeMock();
  const device = await makeDevice();
  const buffer = device.createBuffer({ size: Math.max(bytes.length, 4), usage: GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(buffer, 0, bytes);
  device.queue.submit([]);
  return payloadOf(state, 'writeBuffer');
}

test('writeBuffer puts an ArrayBuffer on as is (the full length)', async () => {
  for (const length of LENGTHS) {
    const bytes = randomBytes(length, length + 1);
    assert.deepEqual(await uploadedByShim(bytes), bytes, `length=${length}`);
  }
});

test('mapAsync returns the ArrayBuffer native gave, undecoded', async () => {
  for (const length of LENGTHS) {
    const bytes = randomBytes(length, length + 7);
    // Native (`LynxWebGPUContext.readBuffer`) puts a Data on and Lynx converts it to an ArrayBuffer.
    const backing = bytes.buffer.slice(0);
    installNativeMock({ readBufferResult: { ok: true, data: backing, byteLength: length } });
    const device = await makeDevice();
    const buffer = device.createBuffer({ size: Math.max(length, 4), usage: GPUBufferUsage.MAP_READ });

    const mapped = await buffer.mapAsync(GPUMapMode.READ);
    assert.ok(mapped instanceof ArrayBuffer, `length=${length}: not an ArrayBuffer`);
    assert.deepEqual(new Uint8Array(mapped), bytes, `length=${length}`);
    // It uses what it was handed as is, without taking a copy.
    assert.equal(buffer.getMappedRange(), mapped);
  }
});

test('reflects a TypedArray byteOffset and the element offset/count when slicing', async () => {
  const backing = new ArrayBuffer(64);
  const full = new Float32Array(backing);
  full.forEach((_, index) => {
    full[index] = index + 0.5;
  });
  const view = new Float32Array(backing, 8, 10); // 10 elements from byteOffset 8

  const state = installNativeMock();
  const device = await makeDevice();
  const buffer = device.createBuffer({ size: 64, usage: GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(buffer, 0, view, 2, 4); // 4 elements from element 2
  device.queue.submit([]);

  const expected = new Uint8Array(backing, 8 + 2 * 4, 4 * 4);
  assert.deepEqual(payloadOf(state, 'writeBuffer'), new Uint8Array(expected));
});

test('writeBuffer copies at call time — changing the source afterwards does not change the payload', async () => {
  // It is a spec contract ("the contents of data are copied"). Commands sit in the queue until submit, so
  // putting one on by reference silently breaks the common pattern of writing several buffers from one
  // array — the constants demo scene really did draw all three polygons on top of each other.
  const uniforms = new Float32Array([1, 2, 3, 4]);
  const state = installNativeMock();
  const device = await makeDevice();
  const bufferA = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST });
  const bufferB = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST });

  device.queue.writeBuffer(bufferA, 0, uniforms);
  uniforms.set([5, 6, 7, 8]);                       // change the same array and write the next buffer
  device.queue.writeBuffer(bufferB, 0, uniforms);
  device.queue.submit([]);

  const writes = commandsOf(state).filter((entry) => entry.op === 'writeBuffer');
  assert.notEqual(writes[0].data, uniforms.buffer, 'the reference went on as is — it must be copied');
  assert.deepEqual(Array.from(new Float32Array(writes[0].data)), [1, 2, 3, 4]);
  assert.deepEqual(Array.from(new Float32Array(writes[1].data)), [5, 6, 7, 8]);
});

test('an ArrayBuffer passed directly is copied at call time too', async () => {
  const backing = randomBytes(16, 3).buffer;
  const state = installNativeMock();
  const device = await makeDevice();
  const buffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(buffer, 0, backing);
  const before = randomBytes(16, 3);
  new Uint8Array(backing).fill(0);
  device.queue.submit([]);

  assert.deepEqual(payloadOf(state, 'writeBuffer'), before);
});

test('an ArrayBuffer and a number array ride as an ArrayBuffer too', async () => {
  const bytes = randomBytes(10, 11);

  const state = installNativeMock();
  const device = await makeDevice();
  const buffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(buffer, 0, bytes.buffer.slice(0, 10));
  device.queue.writeBuffer(buffer, 0, Array.from(bytes));
  device.queue.submit([]);

  const writes = commandsOf(state).filter((entry) => entry.op === 'writeBuffer');
  for (const write of writes) {
    assert.ok(write.data instanceof ArrayBuffer);
    assert.deepEqual(new Uint8Array(write.data), bytes);
  }
});

test('writeTexture puts an ArrayBuffer on too', async () => {
  const texels = randomBytes(64, 21);
  const state = installNativeMock();
  const device = await makeDevice();
  const texture = device.createTexture({
    size: { width: 4, height: 4 },
    format: 'rgba8unorm',
    usage: GPUBufferUsage.COPY_DST,
  });
  device.queue.writeTexture({ texture }, texels, { bytesPerRow: 16 }, { width: 4, height: 4 });
  device.queue.submit([]);

  assert.deepEqual(payloadOf(state, 'writeTexture'), texels);
});

test('mappedAtCreation → unmap puts the initial data on createBuffer as an ArrayBuffer', async () => {
  const bytes = randomBytes(32, 5);
  const state = installNativeMock();
  const device = await makeDevice();
  const buffer = device.createBuffer({ size: 32, usage: GPUBufferUsage.VERTEX, mappedAtCreation: true });
  new Uint8Array(buffer.getMappedRange()).set(bytes);
  buffer.unmap();
  device.queue.submit([]);

  assert.deepEqual(payloadOf(state, 'createBuffer'), bytes);
});
