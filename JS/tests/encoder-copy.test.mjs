/**
 * `copyBufferToBuffer`'s overloads and `clearBuffer` — the call forms the spec fixed.
 *
 * The spec also defines the short form `(source, destination, size?)` and allows `size` to be omitted.
 * Accepting only the 5-argument form would record code that calls it short with the wrong arguments and
 * **no error** (a buffer landing in an integer slot and a NaN offset going out) — so telling the forms apart matters as much as validating values.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { installNativeMock, makeDevice, commandsOf } from './helpers.mjs';

function makeBuffers(device) {
  return {
    source: device.createBuffer({ size: 64, usage: 0x0004 /* COPY_SRC */ }),
    destination: device.createBuffer({ size: 64, usage: 0x0008 /* COPY_DST */ }),
  };
}

function lastCopy(state) {
  return commandsOf(state).filter((command) => command.op === 'copyBufferToBuffer').pop();
}

test('the short form (source, destination) — copies the whole source', async () => {
  const state = installNativeMock();
  const device = await makeDevice();
  const { source, destination } = makeBuffers(device);

  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(source, destination);
  device.queue.submit([encoder.finish()]);

  const command = lastCopy(state);
  assert.equal(command.source, source.id);
  assert.equal(command.destination, destination.id);
  assert.equal(command.sourceOffset, 0);
  assert.equal(command.destinationOffset, 0);
  assert.equal(command.size, 64, 'omitting size means all the remaining bytes of the source');
});

test('the short form plus size', async () => {
  const state = installNativeMock();
  const device = await makeDevice();
  const { source, destination } = makeBuffers(device);

  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(source, destination, 16);
  device.queue.submit([encoder.finish()]);

  const command = lastCopy(state);
  assert.equal(command.size, 16);
  assert.equal(command.destination, destination.id, 'a buffer in the second argument is the destination');
});

test('the long form (source, srcOffset, destination, dstOffset, size)', async () => {
  const state = installNativeMock();
  const device = await makeDevice();
  const { source, destination } = makeBuffers(device);

  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(source, 16, destination, 32, 8);
  device.queue.submit([encoder.finish()]);

  const command = lastCopy(state);
  assert.equal(command.sourceOffset, 16);
  assert.equal(command.destinationOffset, 32);
  assert.equal(command.size, 8);
});

test('omitting size in the long form means the remaining bytes of the source', async () => {
  const state = installNativeMock();
  const device = await makeDevice();
  const { source, destination } = makeBuffers(device);

  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(source, 16, destination, 0);
  device.queue.submit([encoder.finish()]);

  assert.equal(lastCopy(state).size, 48, '64 - 16');
});

test('clearBuffer puts offset and size on as they are, and native fills in an omitted size', async () => {
  const state = installNativeMock();
  const device = await makeDevice();
  const buffer = device.createBuffer({ size: 64, usage: 0x0008 });

  const encoder = device.createCommandEncoder();
  encoder.clearBuffer(buffer, 16, 32);
  encoder.clearBuffer(buffer);
  device.queue.submit([encoder.finish()]);

  const commands = commandsOf(state).filter((command) => command.op === 'clearBuffer');
  assert.equal(commands[0].offset, 16);
  assert.equal(commands[0].size, 32);
  assert.equal(commands[1].offset, 0);
  assert.equal(
    'size' in commands[1], false,
    'size must be left off for native to read it as "to the end" — putting 0 on would clear nothing'
  );
});
