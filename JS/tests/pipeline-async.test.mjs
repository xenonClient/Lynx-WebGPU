/**
 * `createRenderPipelineAsync` / `createComputePipelineAsync` — 실패를 **그 자리에서** 알린다.
 *
 * 동기 판은 명령만 기록하므로 실패가 다음 submit의 오류 배열로 늦게 온다. 비동기 판은
 * 생성을 오류 스코프 두 겹(validation + internal)으로 감싸 즉시 제출하고 결과로 Promise를
 * 푼다. 두 겹인 이유는 파이프라인이 두 종류로 실패하기 때문이다 — 디스크립터 문제는
 * validation, 셰이더 번역·컴파일 실패는 backend(=internal 필터)다. 한 겹만 치면 나머지
 * 절반이 성공으로 풀려 **못 쓰는 파이프라인을 손에 쥔다.**
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { installNativeMock, makeDevice, commandsOf } from './helpers.mjs';

/** popErrorScope 응답을 순서대로 돌려주는 목. */
function mockWithScopes(results) {
  let cursor = 0;
  return installNativeMock({
    executeResult: (payload) => {
      const pops = payload.commands.filter((command) => command.op === 'popErrorScope').length;
      const errorScopes = results.slice(cursor, cursor + pops);
      cursor += pops;
      return { ok: true, commandCount: payload.commands.length, errorScopes };
    },
  });
}

function renderDescriptor(device) {
  const module = device.createShaderModule({ code: '@vertex fn vs() -> @builtin(position) vec4f { return vec4f(0); }' });
  return {
    layout: 'auto',
    vertex: { module, entryPoint: 'vs' },
    fragment: { module, entryPoint: 'fs', targets: [{ format: 'bgra8unorm' }] },
  };
}

test('성공하면 파이프라인으로 풀리고, 스코프 두 겹을 한 배치에 닫는다', async () => {
  const state = mockWithScopes([null, null]);
  const device = await makeDevice();

  const pipeline = await device.createRenderPipelineAsync(renderDescriptor(device));

  assert.ok(pipeline.id > 0, '핸들이 발급돼야 한다');
  const commands = commandsOf(state);
  const ops = commands.map((command) => command.op);
  assert.deepEqual(
    ops.filter((op) => op.indexOf('ErrorScope') >= 0),
    ['pushErrorScope', 'pushErrorScope', 'popErrorScope', 'popErrorScope'],
    '두 겹으로 감싸야 한다'
  );
  assert.deepEqual(
    commands.filter((command) => command.op === 'pushErrorScope').map((command) => command.filter),
    ['validation', 'internal'],
    'internal이 안쪽이어야 backend 오류를 먼저 가져간다'
  );
  assert.equal(state.executeCalls.length, 1, '두 pop이 한 배치에 실려 왕복은 하나여야 한다');
  assert.equal(state.executeCalls[0].present, false, '프레임 중간 제출이므로 present하지 않는다');
});

test('디스크립터 오류(validation)는 GPUPipelineError로 거부한다', async () => {
  // 안쪽(internal) 스코프는 비고, 바깥(validation) 스코프가 잡은 경우.
  mockWithScopes([null, { kind: 'validation', message: '알 수 없는 포맷', path: 'commands[1].fragment' }]);
  const device = await makeDevice();

  await assert.rejects(
    device.createRenderPipelineAsync(renderDescriptor(device)),
    (error) => {
      assert.equal(error.name, 'GPUPipelineError');
      assert.equal(error.reason, 'validation');
      assert.match(error.message, /알 수 없는 포맷/);
      assert.match(error.message, /commands\[1\]\.fragment/, '경로도 함께 와야 진단이 된다');
      return true;
    }
  );
});

test('셰이더 컴파일 실패(backend)도 거부한다 — internal 스코프가 잡는다', async () => {
  mockWithScopes([{ kind: 'backend', message: 'MSL 컴파일 실패' }, null]);
  const device = await makeDevice();

  await assert.rejects(
    device.createRenderPipelineAsync(renderDescriptor(device)),
    (error) => {
      assert.equal(error.reason, 'internal', 'backend는 명세의 internal로 접힌다');
      assert.match(error.message, /MSL 컴파일 실패/);
      return true;
    }
  );
});

test('실패한 파이프라인은 핸들을 남기지 않는다', async () => {
  const state = mockWithScopes([null, { kind: 'validation', message: '거부' }]);
  const device = await makeDevice();

  await assert.rejects(device.createRenderPipelineAsync(renderDescriptor(device)));

  // 거부 뒤 다음 제출에 destroy가 실려야 한다 (레지스트리에 쓰레기를 남기지 않는다).
  device.queue.submit([]);
  const ops = commandsOf(state).map((command) => command.op);
  assert.ok(ops.indexOf('destroy') >= 0, `destroy가 없다: ${ops.join(' ')}`);
});

test('컴퓨트 파이프라인도 같은 계약이다', async () => {
  const state = mockWithScopes([null, null]);
  const device = await makeDevice();
  const module = device.createShaderModule({ code: '@compute @workgroup_size(1) fn main() {}' });

  const pipeline = await device.createComputePipelineAsync({
    layout: 'auto', compute: { module, entryPoint: 'main' },
  });

  assert.ok(pipeline.id > 0);
  const ops = commandsOf(state).map((command) => command.op);
  assert.ok(ops.indexOf('createComputePipeline') >= 0);
});
