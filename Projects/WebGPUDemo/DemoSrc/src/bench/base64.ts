/**
 * 셰임이 쓰던 base64 인코더 — **비교 대상으로만** 남겨 둔 사본이다.
 *
 * 셰임은 이제 `ArrayBuffer`를 그대로 실으므로 이 코드가 런타임 경로에 없다. 벤치에서
 * 새로 짜 넣으면 비교가 공정하지 않으므로, 지워지기 직전 구현(`48343f9`의 룩업 테이블 +
 * 청크 방식)을 **글자 그대로** 옮겨 왔다. 손대지 말 것.
 */

const BASE64_ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'
const BASE64_PAD = 61 // '='

/** 6비트 값 → 문자 코드. */
const BASE64_CODES = (() => {
  const codes = new Uint8Array(64)
  for (let index = 0; index < 64; index += 1) codes[index] = BASE64_ALPHABET.charCodeAt(index)
  return codes
})()

// 텍스처 업로드처럼 수십 KB~수 MB가 지나가는 경로다. 문자열 누적(`+=`)과 문자당
// `indexOf` 스캔은 여기서 바로 병목이 되므로, 룩업 테이블로 문자 코드를 만든 뒤
// `String.fromCharCode`를 청크 단위로 호출해 O(n)으로 처리한다.
export function encodeBase64(bytes: Uint8Array): string {
  const length = bytes.length
  const parts: string[] = []
  const codes: number[] = []
  let index = 0
  const tripleEnd = length - (length % 3)
  while (index < tripleEnd) {
    const a = bytes[index]
    const b = bytes[index + 1]
    const c = bytes[index + 2]
    index += 3
    codes.push(
      BASE64_CODES[a >> 2],
      BASE64_CODES[((a & 0x03) << 4) | (b >> 4)],
      BASE64_CODES[((b & 0x0f) << 2) | (c >> 6)],
      BASE64_CODES[c & 0x3f]
    )
    // apply의 인자 개수 제한을 넘지 않도록 잘라서 문자열로 바꾼다.
    if (codes.length >= 4096) {
      parts.push(String.fromCharCode.apply(null, codes))
      codes.length = 0
    }
  }
  const remainder = length - index
  if (remainder === 1) {
    const a = bytes[index]
    codes.push(BASE64_CODES[a >> 2], BASE64_CODES[(a & 0x03) << 4], BASE64_PAD, BASE64_PAD)
  } else if (remainder === 2) {
    const a = bytes[index]
    const b = bytes[index + 1]
    codes.push(
      BASE64_CODES[a >> 2],
      BASE64_CODES[((a & 0x03) << 4) | (b >> 4)],
      BASE64_CODES[(b & 0x0f) << 2],
      BASE64_PAD
    )
  }
  if (codes.length > 0) parts.push(String.fromCharCode.apply(null, codes))
  return parts.join('')
}
