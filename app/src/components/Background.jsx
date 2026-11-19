// Slowly drifting grid with a soft light that follows the pointer.
import { useEffect, useRef } from "react";

export default function Background() {
  const ref = useRef(null);
  useEffect(() => {
    const c = ref.current, g = c.getContext("2d");
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let raf, t = 0, mx = 0.75, my = 0.1, tx = 0.75, ty = 0.1;
    const D = Math.min(window.devicePixelRatio || 1, 2);
    const fit = () => { c.width = innerWidth * D; c.height = innerHeight * D; };
    const move = (e) => { tx = e.clientX / innerWidth; ty = e.clientY / innerHeight; };
    fit();
    addEventListener("resize", fit);
    addEventListener("pointermove", move, { passive: true });
    const draw = () => {
      t++; mx += (tx - mx) * 0.05; my += (ty - my) * 0.05;
      const w = c.width, h = c.height, s = 32 * D;
      g.clearRect(0, 0, w, h);
      g.strokeStyle = "rgba(255,255,255,0.03)"; g.lineWidth = 1;
      for (let x = reduce ? 0 : (t * 0.12 * D) % s; x < w; x += s) { g.beginPath(); g.moveTo(x, 0); g.lineTo(x, h); g.stroke(); }
      for (let y = 0; y < h; y += s) { g.beginPath(); g.moveTo(0, y); g.lineTo(w, y); g.stroke(); }
      const gr = g.createRadialGradient(mx * w, my * h, 0, mx * w, my * h, Math.max(w, h) * 0.45);
      gr.addColorStop(0, "rgba(124,131,255,0.12)"); gr.addColorStop(1, "rgba(124,131,255,0)");
      g.fillStyle = gr; g.fillRect(0, 0, w, h);
      raf = requestAnimationFrame(draw);
    };
    draw();
    return () => { cancelAnimationFrame(raf); removeEventListener("resize", fit); removeEventListener("pointermove", move); };
  }, []);
  return <canvas ref={ref} className="bg-canvas" aria-hidden="true" />;
}
