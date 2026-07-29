#!/usr/bin/env bash
#
# 최신 버전 태그를 문서의 SPM 버전 표기에 반영한다.
#
# `.claude/settings.json`의 PostToolUse 훅이 `git tag …` 실행 뒤에 부른다.
# 손으로 돌려도 된다 — 하는 일이 멱등이라 몇 번을 돌려도 결과가 같다.
#
#   .claude/hooks/sync-readme-version.sh
#
# 태그가 없거나 이미 최신이면 아무것도 하지 않고 조용히 끝난다.
set -uo pipefail

cd "${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null)}" 2>/dev/null || exit 0

# semver 태그만 본다 (`0.1.0` / `v0.1.0` 둘 다 허용).
latest=$(git tag --list --sort=-v:refname 2>/dev/null |
    grep -E '^v?[0-9]+\.[0-9]+\.[0-9]+$' | head -1)
[ -n "$latest" ] || exit 0
version=${latest#v}

# 이 패키지를 가리키는 `.package(url: …Lynx-WebGPU…, from: "x.y.z")` 줄만 고친다.
targets="README.md docs/LYNX-INTEGRATION.md"
changed=""
for file in $targets; do
    [ -f "$file" ] || continue
    before=$(cat "$file")
    after=$(sed -E "/Lynx-WebGPU/ s/from: \"[0-9]+\.[0-9]+\.[0-9]+\"/from: \"${version}\"/" "$file")
    if [ "$before" != "$after" ]; then
        printf '%s\n' "$after" >"$file"
        changed="${changed}${changed:+, }${file}"
    fi
done

[ -n "$changed" ] || exit 0
printf '{"systemMessage":"SPM 버전 표기를 %s 로 갱신했다 — %s"}\n' "$version" "$changed"
