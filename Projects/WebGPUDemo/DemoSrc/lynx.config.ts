import { defineConfig } from '@lynx-js/rspeedy'
import { pluginReactLynx } from '@lynx-js/react-rsbuild-plugin'

// One entry per demo scene → dist/<scene>.lynx.bundle.
// `npm run sync` copies them to ../Resources/ after the build (the host app picks one with -demo <name>).
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
      spec: './src/spec/index.tsx',
      images: './src/images/index.tsx',
      contracts: './src/contracts/index.tsx',
      threelab: './src/threelab/index.tsx',
    },
    // PrimJS has no web globals such as self/performance/navigator, and a `globalThis.X = …` assignment is
    // not reflected in bare identifier resolution. So the names web libraries (three.js and the like) use
    // are swapped at compile time for the lynx* globals the shim (webgpu.js) installs at load time.
    // Declared bindings (the shim's own `export const GPUBufferUsage`, etc.) are not substituted.
    // The full recipe: docs/JS-AUTHORING.md §10.
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
