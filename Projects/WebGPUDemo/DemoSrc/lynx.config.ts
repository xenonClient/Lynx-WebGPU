import { defineConfig } from '@lynx-js/rspeedy'
import { pluginReactLynx } from '@lynx-js/react-rsbuild-plugin'

// 데모 씬 1개당 entry 1개 → dist/<scene>.lynx.bundle 산출.
// `npm run sync` 가 빌드 후 ../Resources/ 로 복사한다 (호스트 앱이 -demo <name> 으로 고른다).
export default defineConfig({
  source: {
    entry: {
      triangle: './src/triangle/index.tsx',
      cube: './src/cube/index.tsx',
      particles: './src/particles/index.tsx',
      texture: './src/texture/index.tsx',
      dynamic: './src/dynamic/index.tsx',
      blending: './src/blending/index.tsx',
      stencil: './src/stencil/index.tsx',
      gpudriven: './src/gpudriven/index.tsx',
      bundle: './src/bundle/index.tsx',
      query: './src/query/index.tsx',
      readback: './src/readback/index.tsx',
      constants: './src/constants/index.tsx',
      msl: './src/msl/index.tsx',
      interactive: './src/interactive/index.tsx',
      wgsl: './src/wgsl/index.tsx',
      arraybuffer: './src/arraybuffer/index.tsx',
      bench: './src/bench/index.tsx',
      hdr: './src/hdr/index.tsx',
      scrollpass: './src/scrollpass/index.tsx',
      three: './src/three/index.tsx',
    },
    // PrimJS에는 self/performance/navigator 같은 웹 전역이 없고, `globalThis.X = …` 대입은
    // bare 식별자 해석에 반영되지 않는다. 그래서 웹 라이브러리(Three.js 등)가 쓰는 이름을
    // shim(webgpu.js)이 로드 시점에 얹는 lynx* 전역으로 컴파일 타임에 바꿔치기한다.
    // 선언된 바인딩(shim 자신의 `export const GPUBufferUsage` 등)은 치환되지 않는다.
    // 전체 조리법: docs/JS-AUTHORING.md §10.
    define: {
      self: 'globalThis',
      performance: 'globalThis.lynxPerformance',
      navigator: 'globalThis.lynxNavigator',
      GPUBufferUsage: 'globalThis.lynxGPUBufferUsage',
      GPUTextureUsage: 'globalThis.lynxGPUTextureUsage',
      GPUShaderStage: 'globalThis.lynxGPUShaderStage',
      GPUColorWrite: 'globalThis.lynxGPUColorWrite',
      GPUMapMode: 'globalThis.lynxGPUMapMode',
    },
  },
  output: {
    filenameHash: false,
  },
  plugins: [pluginReactLynx()],
})
