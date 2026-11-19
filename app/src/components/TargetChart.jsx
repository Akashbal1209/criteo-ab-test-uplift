// "Who to target": share of extra visits captured when the most responsive people see ads first.
import { useEffect, useRef } from "react";
import Swap from "./Swap.jsx";
import { capturedShare } from "../lib/metrics.js";
import { useTech } from "../tech.js";

export default function TargetChart({ curves, models, selX }) {
  const tech = useTech();
  const ref = useRef(null);
  const state = useRef({ hov: null, dp: 0, selX, tech });
  state.current.selX = selX;
  state.current.tech = tech;

  useEffect(() => {
    const c = ref.current, g = c.getContext("2d");
    const D = Math.min(window.devicePixelRatio || 1, 2);
    const S = state.current;
    const model = (x) => capturedShare(curves.s_learner, x);
    const act = (x) => capturedShare(curves.activity, x);
    const at20 = Math.round(model(0.2) * 100);
    let raf;
    const fit = () => { const r = c.getBoundingClientRect(); c.width = r.width * D; c.height = r.height * D; };
    const ro = new ResizeObserver(fit); ro.observe(c); fit();
    const move = (e) => { const r = c.getBoundingClientRect(); S.hov = Math.max(0, Math.min(1, (e.clientX - r.left - 34) / (r.width - 46))); };
    const leave = () => { S.hov = null; };
    c.addEventListener("pointermove", move); c.addEventListener("pointerleave", leave);
    const draw = () => {
      const w = c.width, h = c.height, L = 34 * D, R = 12 * D, T = 10 * D, B = 22 * D;
      const sx = (v) => L + v * (w - L - R), sy = (v) => T + (1 - v) * (h - T - B);
      g.clearRect(0, 0, w, h);
      g.font = `${10.5 * D}px Inter, sans-serif`; g.lineWidth = D;
      [0, 0.5, 1].forEach((v) => {
        g.strokeStyle = "#1c1d20"; g.beginPath(); g.moveTo(L, sy(v)); g.lineTo(w - R, sy(v)); g.stroke();
        g.fillStyle = "#5d626b"; g.textAlign = "right"; g.fillText(`${v * 100}%`, L - 6 * D, sy(v) + 3.5 * D);
        g.textAlign = v === 0 ? "left" : v === 1 ? "right" : "center";
        g.fillText(`${v * 100}%${v === 1 ? " of people" : ""}`, sx(v), h - 5 * D);
      });
      S.dp = Math.min(1, S.dp + 0.025);
      const n = Math.max(1, Math.floor(100 * S.dp));
      const pts = (f) => { const a = [[sx(0), sy(0)]]; for (let i = 1; i <= n; i++) a.push([sx(i / 100), sy(f(i / 100))]); return a; };
      const mp = pts(model);
      const gr = g.createLinearGradient(0, T, 0, h - B);
      gr.addColorStop(0, "rgba(124,131,255,.28)"); gr.addColorStop(1, "rgba(124,131,255,0)");
      g.fillStyle = gr; g.beginPath(); mp.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y)));
      g.lineTo(sx(n / 100), sy(0)); g.closePath(); g.fill();
      g.setLineDash([4 * D, 4 * D]); g.strokeStyle = "#33363c"; g.lineWidth = 1.2 * D;
      g.beginPath(); g.moveTo(sx(0), sy(0)); g.lineTo(sx(S.dp), sy(S.dp)); g.stroke(); g.setLineDash([]);
      g.strokeStyle = "#4a4e57"; g.lineWidth = 1.3 * D; g.beginPath(); pts(act).forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y))); g.stroke();
      g.strokeStyle = "#a5a9ff"; g.lineWidth = 2.2 * D; g.beginPath(); mp.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y))); g.stroke();
      if (S.dp >= 1) {
        g.textAlign = "left"; g.font = `${10.5 * D}px Inter, sans-serif`;
        g.fillStyle = "#a5a9ff"; g.fillText(S.tech ? "S-Learner" : "Our model", sx(0.55), sy(model(0.55)) + 15 * D);
        g.fillStyle = "#5d626b"; g.fillText("Random", sx(0.7), sy(0.7) + 14 * D);
        g.fillStyle = "#ededef"; g.beginPath(); g.arc(sx(0.2), sy(model(0.2)), 3.5 * D, 0, 7); g.fill();
        g.font = `500 ${11 * D}px Inter, sans-serif`;
        g.fillText(S.tech ? `uplift share ${at20}% @ 20%` : `Top 20% → ${at20}% of extra visits`, sx(0.2) + 8 * D, sy(model(0.2)) + 4 * D);
        if (S.selX !== null) { g.fillStyle = "#fbbf24"; g.beginPath(); g.arc(sx(S.selX), sy(model(S.selX)), 4.5 * D, 0, 7); g.fill(); }
      }
      if (S.hov !== null) {
        const v = Math.max(0.01, S.hov);
        g.strokeStyle = "#33363c"; g.beginPath(); g.moveTo(sx(v), T); g.lineTo(sx(v), h - B); g.stroke();
        g.fillStyle = "#a5a9ff"; g.beginPath(); g.arc(sx(v), sy(model(v)), 4 * D, 0, 7); g.fill();
        const txt = `Top ${Math.round(v * 100)}% → ${Math.round(model(v) * 100)}%`;
        g.font = `500 ${11.5 * D}px Inter, sans-serif`;
        const tw = g.measureText(txt).width + 14 * D, x0 = Math.min(Math.max(sx(v) - tw / 2, L), w - R - tw);
        g.fillStyle = "#16171a"; g.strokeStyle = "#2a2c31"; g.beginPath(); g.roundRect(x0, T, tw, 22 * D, 6 * D); g.fill(); g.stroke();
        g.fillStyle = "#ededef"; g.textAlign = "left"; g.fillText(txt, x0 + 7 * D, T + 15 * D);
      }
      raf = requestAnimationFrame(draw);
    };
    draw();
    return () => { cancelAnimationFrame(raf); ro.disconnect(); c.removeEventListener("pointermove", move); c.removeEventListener("pointerleave", leave); };
  }, [curves]);

  const m = models.s_learner, b = models.activity;
  return (
    <section className="card" aria-label="Who to target">
      <Swap as="div" className="lbl" s="Who to target" t="Qini curve (test set)" />
      <Swap as="div" className="one tight" s="Show ads to the most responsive people first"
        t={`S-Learner ${m.qini.toFixed(3)} [${m.qini_ci[0].toFixed(3)}, ${m.qini_ci[1].toFixed(3)}] · activity ${b.qini.toFixed(3)}`} />
      <div className="chart-box"><canvas ref={ref} role="img" aria-label="Share of extra visits captured by targeting share" /></div>
    </section>
  );
}
