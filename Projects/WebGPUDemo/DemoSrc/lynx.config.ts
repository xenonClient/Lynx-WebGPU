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
    },
  },
  output: {
    filenameHash: false,
  },
  plugins: [pluginReactLynx()],
})
