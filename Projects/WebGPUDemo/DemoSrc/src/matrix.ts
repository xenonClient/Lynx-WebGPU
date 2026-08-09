/**
 * A 4×4 matrix — stored **column-major**. The layout matches WGSL's `mat4x4<f32>`, so a
 * `Float32Array` can be written straight into a uniform buffer.
 *
 * Element (row r, col c) lives at `m[c * 4 + r]`.
 */
export type Mat4 = Float32Array

export function identity(): Mat4 {
  const out = new Float32Array(16)
  out[0] = 1
  out[5] = 1
  out[10] = 1
  out[15] = 1
  return out
}

export function multiply(a: Mat4, b: Mat4): Mat4 {
  const out = new Float32Array(16)
  for (let col = 0; col < 4; col++) {
    for (let row = 0; row < 4; row++) {
      let sum = 0
      for (let k = 0; k < 4; k++) sum += a[k * 4 + row] * b[col * 4 + k]
      out[col * 4 + row] = sum
    }
  }
  return out
}

/** Multiplies several from the left — `multiplyAll(P, V, M)` = P·V·M. */
export function multiplyAll(...matrices: Mat4[]): Mat4 {
  return matrices.reduce((accumulated, next) => multiply(accumulated, next))
}

/** WebGPU clip space (z 0~1, right-handed, looking down -Z). */
export function perspective(fovY: number, aspect: number, near: number, far: number): Mat4 {
  const f = 1 / Math.tan(fovY / 2)
  const out = new Float32Array(16)
  out[0] = f / aspect
  out[5] = f
  out[10] = far / (near - far)
  out[11] = -1
  out[14] = (far * near) / (near - far)
  return out
}

export function translation(x: number, y: number, z: number): Mat4 {
  const out = identity()
  out[12] = x
  out[13] = y
  out[14] = z
  return out
}

export function scaling(x: number, y: number, z: number): Mat4 {
  const out = new Float32Array(16)
  out[0] = x
  out[5] = y
  out[10] = z
  out[15] = 1
  return out
}

export function rotationX(angle: number): Mat4 {
  const c = Math.cos(angle)
  const s = Math.sin(angle)
  const out = identity()
  out[5] = c
  out[6] = s
  out[9] = -s
  out[10] = c
  return out
}

export function rotationY(angle: number): Mat4 {
  const c = Math.cos(angle)
  const s = Math.sin(angle)
  const out = identity()
  out[0] = c
  out[2] = -s
  out[8] = s
  out[10] = c
  return out
}

/** Transforms one point by a matrix, perspective divide included (for shadow position math). */
export function project(matrix: Mat4, x: number, y: number, z: number) {
  const clip = [0, 0, 0, 0]
  for (let row = 0; row < 4; row++) {
    clip[row] = matrix[row] * x + matrix[4 + row] * y + matrix[8 + row] * z + matrix[12 + row]
  }
  const w = clip[3] === 0 ? 1 : clip[3]
  return { x: clip[0] / w, y: clip[1] / w, z: clip[2] / w }
}
