import { useEffect, useRef, useState } from "react";

// PDF-style zoom + pan wrapper, vanilla implementation (no library).
//
// Why a non-passive wheel listener via useEffect: React's synthetic `onWheel`
// is registered as passive in React 17+, so calling `e.preventDefault()`
// silently fails and the page scrolls instead of zooming. We bypass that by
// attaching the listener with `{ passive: false }` ourselves.

const MIN_SCALE = 0.2;
const MAX_SCALE = 8;
const ZOOM_PER_TICK = 1.1; // wheel step factor

export default function ZoomPanVanilla({ children, height = "70vh" }) {
  const containerRef = useRef(null);
  const [scale, setScale] = useState(1);
  const [tx, setTx] = useState(0);
  const [ty, setTy] = useState(0);
  const [dragging, setDragging] = useState(false);
  // Drag origin (pointer position + translate at mousedown). Stored in a ref
  // so the move handler reads the latest values without re-binding.
  const drag = useRef({ x: 0, y: 0, tx: 0, ty: 0 });

  // Cursor-anchored wheel zoom: the content point under the cursor stays put.
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const onWheel = (e) => {
      e.preventDefault();
      const rect = el.getBoundingClientRect();
      const cx = e.clientX - rect.left;
      const cy = e.clientY - rect.top;
      const factor = e.deltaY < 0 ? ZOOM_PER_TICK : 1 / ZOOM_PER_TICK;
      setScale((s) => {
        const next = Math.max(MIN_SCALE, Math.min(MAX_SCALE, s * factor));
        const ratio = next / s;
        setTx((x) => cx - (cx - x) * ratio);
        setTy((y) => cy - (cy - y) * ratio);
        return next;
      });
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, []);

  const onPointerDown = (e) => {
    // Ignore drags initiated on interactive children (buttons in the legend
    // etc.) so they still get clicks.
    if (e.target.closest("button")) return;
    setDragging(true);
    drag.current = { x: e.clientX, y: e.clientY, tx, ty };
    e.currentTarget.setPointerCapture(e.pointerId);
  };
  const onPointerMove = (e) => {
    if (!dragging) return;
    setTx(drag.current.tx + (e.clientX - drag.current.x));
    setTy(drag.current.ty + (e.clientY - drag.current.y));
  };
  const onPointerUp = (e) => {
    if (!dragging) return;
    setDragging(false);
    e.currentTarget.releasePointerCapture(e.pointerId);
  };
  const reset = () => {
    setScale(1);
    setTx(0);
    setTy(0);
  };

  return (
    <div
      ref={containerRef}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
      style={{
        position: "relative",
        overflow: "hidden",
        height,
        border: "1px solid #eee",
        cursor: dragging ? "grabbing" : "grab",
        touchAction: "none",
        userSelect: "none",
      }}
    >
      <div
        style={{
          transform: `translate(${tx}px, ${ty}px) scale(${scale})`,
          transformOrigin: "0 0",
          display: "inline-block",
        }}
      >
        {children}
      </div>
      <button
        onClick={reset}
        title="reset zoom + pan"
        style={{
          position: "absolute",
          top: 8,
          right: 8,
          fontFamily: "'JetBrains Mono','Fira Mono','Consolas',monospace",
          fontSize: 10,
          padding: "3px 8px",
          background: "#fff",
          border: "1px solid #ccc",
          cursor: "pointer",
        }}
      >
        reset · {scale.toFixed(2)}×
      </button>
    </div>
  );
}
