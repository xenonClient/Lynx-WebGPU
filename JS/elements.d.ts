/**
 * `<webgpu-canvas>` 커스텀 엘리먼트의 TSX 타입 선언.
 *
 * rspeedy 프로젝트의 `src/` 아래로 복사하면 TSX에서 태그를 쓸 수 있다.
 */

interface LynxTouchLike {
  touches: Array<{ identifier: number; x: number; y: number; pageX: number; pageY: number }>;
  changedTouches: Array<{ identifier: number; x: number; y: number }>;
  detail: { x: number; y: number };
  stopPropagation: () => void;
  preventDefault: () => void;
}

export interface WebGPUCanvasProps {
  /** JS가 `gpu.getCanvasContext(id)` / `configure({canvas})`에서 지목할 이름. 페이지 안에서 유일해야 한다. */
  'canvas-id': string;
  /** CSS px → 드로어블 픽셀 배율. 생략하면 화면 배율을 쓴다 (성능을 위해 1로 낮출 수 있다). */
  'pixel-ratio'?: number;
  /** 드로어블 픽셀 크기가 바뀔 때. 투영행렬·뷰포트를 다시 계산할 지점이다. */
  bindcanvasresize?: (event: {
    detail: { width: number; height: number; pixelRatio: number };
  }) => void;
  /**
   * 터치 입력은 **Lynx 표준 이벤트**를 그대로 쓴다 — 이 엘리먼트가 따로 만든 이벤트가 아니다.
   * 그래야 히트 테스트·버블링·`catch` 접두사·스크롤 제스처가 다른 엘리먼트와 똑같이 동작한다.
   * `touches[].x/y`는 **엘리먼트 기준 CSS px**이므로, 캔버스 크기로 나누면 0~1이 된다.
   */
  bindtouchstart?: (event: LynxTouchLike) => void;
  bindtouchmove?: (event: LynxTouchLike) => void;
  bindtouchend?: (event: LynxTouchLike) => void;
  bindtouchcancel?: (event: LynxTouchLike) => void;
  bindtap?: (event: LynxTouchLike) => void;
  className?: string;
  style?: string | Record<string, string | number>;
  id?: string;
}

declare module '@lynx-js/types' {
  interface IntrinsicElements {
    'webgpu-canvas': WebGPUCanvasProps;
  }
}
