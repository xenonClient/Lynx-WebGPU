#!/bin/zsh
#
# 회귀 테스트 — 작업이 끝날 때 자동으로 돈다 (`.claude/settings.json`의 Stop 훅).
#
# 왜 훅인가: 이 저장소는 층이 넷이다 (Swift 엔진 · JS shim · 트랜스파일러 · 데모 번들).
# 한 층만 고치고 넘어가면 다른 층이 조용히 어긋난다 — 실제로 `webgpu.js`만 고치고
# `webgpu.d.ts`를 안 뽑아 공개 선언이 뒤처진 적이 여러 번 있었다.
#
# 여기서 도는 것 (빠른 순):
#   1. JS 타입 검사 — JSDoc이 빠지면 여기서 걸린다
#   2. JS 선언 드리프트 — `webgpu.d.ts`가 구현보다 뒤처졌는지
#   3. 데모 사본 드리프트 — `DemoSrc/src/webgpu.js`가 원본과 다른지
#   4. JS 단위 테스트
#   5. Swift 테스트 (GPU 포함)
#
# **데모 앱 빌드와 시뮬레이터 실행은 넣지 않았다.** 몇 분이 걸려 매 턴 돌리기엔 무겁다 —
# 브리지나 씬을 고쳤으면 `docs/TESTING.md` §8의 명령을 직접 돌릴 것.
#
# 건너뛰려면 `LYNXWEBGPU_SKIP_REGRESSION=1`.

set -uo pipefail
cd "${CLAUDE_PROJECT_DIR:-$(dirname "$0")/../..}" || exit 0

if [[ -n "${LYNXWEBGPU_SKIP_REGRESSION:-}" ]]; then
  exit 0
fi

# 변경이 없으면 돌리지 않는다 — 대화만 한 턴에서 5초를 쓰지 않게.
if [[ -z "$(git status --porcelain 2>/dev/null)" ]]; then
  exit 0
fi

failures=()

step() {
  local label="$1"; shift
  local output
  if ! output="$("$@" 2>&1)"; then
    failures+=("$label")
    print -r -- "── $label 실패 ──"
    print -r -- "$output" | tail -20
  fi
}

# 1~2. 타입 검사와 선언 드리프트
step "JS 타입 검사 (npm run typecheck)" npm --prefix JS run typecheck

if command -v git >/dev/null; then
  before="$(shasum JS/webgpu.d.ts 2>/dev/null | cut -d' ' -f1)"
  npm --prefix JS run types >/dev/null 2>&1
  after="$(shasum JS/webgpu.d.ts 2>/dev/null | cut -d' ' -f1)"
  if [[ "$before" != "$after" ]]; then
    failures+=("webgpu.d.ts 가 구현보다 뒤처져 있었다 (지금 다시 뽑아 두었다 — 커밋할 것)")
  fi
fi

# 3. 데모가 쓰는 shim 사본
for file in webgpu.js webgpu.d.ts; do
  if [[ -f "Projects/WebGPUDemo/DemoSrc/src/$file" ]] \
     && ! cmp -s "JS/$file" "Projects/WebGPUDemo/DemoSrc/src/$file"; then
    failures+=("데모 사본이 원본과 다르다: $file (cp JS/$file Projects/WebGPUDemo/DemoSrc/src/)")
  fi
done

# 4~5. 테스트
step "JS 테스트 (npm test)" npm --prefix JS test
step "Swift 테스트 (swift test)" swift test

if (( ${#failures[@]} > 0 )); then
  print -r -- ""
  print -r -- "회귀 테스트 실패 ${#failures[@]}건:"
  for entry in "${failures[@]}"; do print -r -- "  · $entry"; done
  exit 2   # exit 2 = 이 내용을 모델에게 되돌려 준다
fi

print -r -- "회귀 테스트 통과 (JS 타입·선언·사본 · JS 테스트 · Swift 테스트)"
exit 0
