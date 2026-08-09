/**
 * The base64 encoder the shim used to use — a copy kept **purely for comparison**.
 *
 * The shim now puts an `ArrayBuffer` on directly, so this code is not on the runtime path. Writing a fresh
 * one for the benchmark would make the comparison unfair, so the implementation right before deletion
 * (`48343f9`'s lookup table plus chunking) was moved here **verbatim**. Do not touch it.
 */

const BASE64_ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'
const BASE64_PAD = 61 // '='

/** A 6-bit value → a character code. */
const BASE64_CODES = (() => {
  const codes = new Uint8Array(64)
  for (let index = 0; index < 64; index += 1) codes[index] = BASE64_ALPHABET.charCodeAt(index)
  return codes
})()

// This is a path where tens of KB to several MB pass through, as with a texture upload. String accumulation
// (`+=`) and a per-character `indexOf` scan become the bottleneck right here, so character codes are built
// from a lookup table and `String.fromCharCode` is called per chunk, making it O(n).
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
    // Sliced before converting to a string, so apply's argument count limit is not exceeded.
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
