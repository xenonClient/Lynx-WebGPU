/**
 * TSX type declarations for the `<webgpu-canvas>` custom element.
 *
 * Copy it under an rspeedy project's `src/` and the tag becomes usable from TSX.
 */

interface LynxTouchLike {
  touches: Array<{ identifier: number; x: number; y: number; pageX: number; pageY: number }>;
  changedTouches: Array<{ identifier: number; x: number; y: number }>;
  detail: { x: number; y: number };
  stopPropagation: () => void;
  preventDefault: () => void;
}

export interface WebGPUCanvasProps {
  /** The name JS points at in `gpu.getCanvasContext(id)` / `configure({canvas})`. It must be unique within the page. */
  'canvas-id': string;
  /** CSS px → drawable pixel scale. Omitted, the screen scale is used (it can be lowered to 1 for performance). */
  'pixel-ratio'?: number;
  /**
   * UIKit touch passthrough. Off by default — it covers what is beneath, like a canvas on the web.
   *
   * Turn it on when the canvas **overlaps a native-view-based element (`<scroll-view>` and the like) as a
   * sibling above it**, and that element's gestures (a scroll pan, say) pass down through the canvas. The
   * canvas's own Lynx events (`bindtouchstart` …) keep arriving — if a passed-through gesture wins, a `bindtouchcancel` follows.
   * When the scroll view is an **ancestor** of the canvas, scrolling works without this prop.
   */
  'passthrough-touches'?: boolean;
  /** When the drawable pixel size changes. This is where the projection matrix and viewport get recomputed. */
  bindcanvasresize?: (event: {
    detail: { width: number; height: number; pixelRatio: number };
  }) => void;
  /**
   * Touch input uses **the Lynx standard events** as they are — these are not events this element invented.
   * That is what makes hit testing, bubbling, the `catch` prefix and scroll gestures behave exactly as on any other element.
   * `touches[].x/y` are **CSS px relative to the element**, so dividing by the canvas size gives 0~1.
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
