/**
 * `<webgpu-canvas>` 커스텀 엘리먼트의 TSX 타입 선언.
 *
 * rspeedy 프로젝트의 `src/` 아래로 복사하면 TSX에서 태그를 쓸 수 있다.
 */

export interface WebGPUCanvasProps {
  /** JS가 `gpu.getCanvasContext(id)` / `configure({canvas})`에서 지목할 이름. 페이지 안에서 유일해야 한다. */
  'canvas-id': string;
  /** CSS px → 드로어블 픽셀 배율. 생략하면 화면 배율을 쓴다 (성능을 위해 1로 낮출 수 있다). */
  'pixel-ratio'?: number;
  /** 드로어블 픽셀 크기가 바뀔 때. 투영행렬·뷰포트를 다시 계산할 지점이다. */
  bindcanvasresize?: (event: {
    detail: { width: number; height: number; pixelRatio: number };
  }) => void;
  className?: string;
  style?: string | Record<string, string | number>;
  id?: string;
}

declare module '@lynx-js/types' {
  interface IntrinsicElements {
    'webgpu-canvas': WebGPUCanvasProps;
  }
}
