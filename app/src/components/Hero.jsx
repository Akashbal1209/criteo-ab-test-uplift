import { useEffect, useState } from "react";
import Swap from "./Swap.jsx";
import { fmtPP, fmtPct } from "../lib/metrics.js";

export default function Hero({ e }) {
  const v = e.visit;
  const perK = Math.round(v.itt * 1000);
  const withAds = Math.round(v.rate_t * 1000), noAds = Math.round(v.rate_c * 1000);
  const [grow, setGrow] = useState(false);
  useEffect(() => { const id = setTimeout(() => setGrow(true), 120); return () => clearTimeout(id); }, []);
  const max = Math.max(withAds, noAds) * 1.02;
  return (
    <section className="card hero" aria-label="Main result">
      <div>
        <Swap as="div" className="lbl" s="Extra visitors per 1,000" t="Lift in visit rate" />
        <Swap as="div" className="big hero-num" s={`+${perK}`} t={fmtPP(v.itt)} />
        <Swap as="div" className="one" s="caused by the ads"
          t={`95% CI [${(v.ci[0] * 100).toFixed(2)}, ${(v.ci[1] * 100).toFixed(2)}] · adjusted ${fmtPP(v.adjusted)}`} />
      </div>
      <div className="bars">
        {[["With ads", "Treatment", withAds, v.rate_t, "var(--ad)"], ["Without ads", "Control", noAds, v.rate_c, "var(--noad)"]].map(
          ([s, t, n, rate, col]) => (
            <div className="bar" key={s}>
              <Swap s={s} t={t} style={{ color: col }} />
              <div className="track"><div className="fill" style={{ width: grow ? `${(n / max) * 100}%` : 0, background: col }} /></div>
              <Swap className="val" s={String(n)} t={fmtPct(rate)} />
            </div>
          )
        )}
        <Swap as="div" className="one" s="Visitors out of every 1,000 people" t="Visit rate by randomized group" />
      </div>
    </section>
  );
}
