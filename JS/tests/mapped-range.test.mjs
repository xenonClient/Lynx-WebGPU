/**
 * `getMappedRange(offset, size)` — partial mapping.
 *
 * It used to **silently ignore** the arguments and always return the whole thing. Code expecting a part
 * would read without the offset and write wrong values, and by the time that shows on screen the cause is hard to find.
 *
 * In JS an `ArrayBuffer` cannot point into part of another `ArrayBuffer`, so the range is handed over as a
 * copy and written back in `unmap()` — **whether what you wrote survives** is the most important contract here.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { installNativeMock, makeDevice, commandsOf } from './helpers.mjs';

test('respects the offset and size and gives only that range', async () => {
  const backing = new Uint8Array(32);
  backing.forEach((_, index) => { backing[index] = index; });
  installNativeMock({ readBufferResult: { ok: true, data: backing.buffer } });
  const device = await makeDevice();
  const buffer = device.createBuffer({ size: 32, usage: 0x0001 });
  await buffer.mapAsync();

  const middle = new Uint8Array(buffer.getMappedRange(8, 8));

  assert.equal(middle.length, 8);
  assert.deepEqual(Array.from(middle), [8, 9, 10, 11, 12, 13, 14, 15]);
});

test('what was written into a range comes back at unmap', async () => {
  const state = installNativeMock();
  const device = await makeDevice();
  const buffer = device.createBuffer({ size: 32, usage: 0x0020, mappedAtCreation: true });

  // Take only the last 8 bytes and fill them — being a copy, they vanish silently without the write-back.
  new Uint8Array(buffer.getMappedRange(24, 8)).fill(0xab);
  buffer.unmap();
  device.queue.submit([]);

  const unmap = commandsOf(state).find((command) => command.op === 'unmapBuffer');
  const bytes = new Uint8Array(unmap.data);
  assert.equal(bytes.length, 32);
  assert.deepEqual(Array.from(bytes.slice(24)), Array(8).fill(0xab), 'what was written vanished');
  assert.deepEqual(Array.from(bytes.slice(0, 8)), Array(8).fill(0), 'the untouched part is unchanged');
});

test('requesting the whole thing first gives the mapping itself with no copy', async () => {
  const backing = new ArrayBuffer(16);
  installNativeMock({ readBufferResult: { ok: true, data: backing } });
  const device = await makeDevice();
  const buffer = device.createBuffer({ size: 16, usage: 0x0001 });
  const mapped = await buffer.mapAsync();

  assert.equal(buffer.getMappedRange(), mapped, 'it saves one copy on a large buffer');
});

test('keeps the spec alignment rules (offset a multiple of 8, an explicit size a multiple of 4)', async () => {
  installNativeMock();
  const device = await makeDevice();
  const buffer = device.createBuffer({ size: 32, usage: 0x0020, mappedAtCreation: true });

  assert.throws(() => buffer.getMappedRange(4), (error) => {
    assert.equal(error.name, 'OperationError');
    assert.match(error.message, /multiple of 8/);
    return true;
  });
  assert.throws(() => buffer.getMappedRange(0, 6), /multiple of 4/);
});

test('rejects overlapping ranges', async () => {
  installNativeMock();
  const device = await makeDevice();
  const buffer = device.createBuffer({ size: 32, usage: 0x0020, mappedAtCreation: true });

  buffer.getMappedRange(0, 16);
  assert.throws(() => buffer.getMappedRange(8, 16), /overlaps/);
  // Not overlapping is fine.
  assert.doesNotThrow(() => buffer.getMappedRange(16, 16));
});

test('rejects going past the end', async () => {
  installNativeMock();
  const device = await makeDevice();
  const buffer = device.createBuffer({ size: 16, usage: 0x0020, mappedAtCreation: true });

  assert.throws(() => buffer.getMappedRange(8, 16), /exceeds/);
});

test('a mapping that is not a multiple of 4 can still be read whole', async () => {
  // The mapping size is the native buffer size, so even a 3-byte one is normal —
  // requiring a multiple of 4 for an omitted size would make such a buffer unreadable.
  const backing = new Uint8Array([1, 2, 3]).buffer;
  installNativeMock({ readBufferResult: { ok: true, data: backing } });
  const device = await makeDevice();
  const buffer = device.createBuffer({ size: 3, usage: 0x0001 });
  await buffer.mapAsync();

  assert.equal(buffer.getMappedRange().byteLength, 3);
});

test('offsets are buffer offsets, not offsets into a partial mapping', async () => {
  // Native hands back just the mapped range [16, 32) of a 64-byte buffer.
  const backing = new Uint8Array(16);
  backing.forEach((_, index) => { backing[index] = 16 + index; });
  installNativeMock({ readBufferResult: { ok: true, data: backing.buffer } });
  const device = await makeDevice();
  const buffer = device.createBuffer({ size: 64, usage: 0x0001 });
  await buffer.mapAsync(1 /* READ */, 16, 16);

  assert.throws(() => buffer.getMappedRange(0, 16), /exceeds the mapping/, 'the range before the mapping');
  assert.throws(() => buffer.getMappedRange(24, 16), /exceeds the mapping/, 'the range past the mapping');
  const tail = new Uint8Array(buffer.getMappedRange(24, 8));
  assert.deepEqual(Array.from(tail), [24, 25, 26, 27, 28, 29, 30, 31]);
  const head = new Uint8Array(buffer.getMappedRange(16, 8));
  assert.deepEqual(Array.from(head), [16, 17, 18, 19, 20, 21, 22, 23]);
});

test('the whole partial mapping comes back as the mapping itself', async () => {
  const backing = new ArrayBuffer(16);
  installNativeMock({ readBufferResult: { ok: true, data: backing } });
  const device = await makeDevice();
  const buffer = device.createBuffer({ size: 64, usage: 0x0001 });
  const mapped = await buffer.mapAsync(1 /* READ */, 16, 16);

  assert.equal(buffer.getMappedRange(16, 16), mapped);
});

test('an omitted size runs to the end of the buffer, not of a partial mapping (spec)', async () => {
  installNativeMock({ readBufferResult: { ok: true, data: new ArrayBuffer(16) } });
  const device = await makeDevice();
  const buffer = device.createBuffer({ size: 64, usage: 0x0001 });
  await buffer.mapAsync(1 /* READ */, 16, 16);

  // [16, 64) is past the mapping [16, 32) — a browser rejects it, so this does too.
  assert.throws(() => buffer.getMappedRange(16), /exceeds the mapping/);
  assert.throws(() => buffer.getMappedRange(), /exceeds the mapping/);
});
