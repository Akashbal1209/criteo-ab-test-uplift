// "Should this person see the ad?" — scores one user with the S-Learner running in the browser.
import { useEffect, useMemo, useState } from "react";
import Swap from "./Swap.jsx";
import { scoreUsers } from "../lib/scorer.js";
import { fmtPP, fmtPrice, percentileOfValue, valueAtPercentile } from "../lib/metrics.js";
import { useTech } from "../tech.js";

const perThousand = (u) => {
  const v = u * 1000;
  if (Math.abs(v) < 0.05) return "≈ 0 / 1,000";
  const txt = Math.abs(v) < 10 ? Math.abs(v).toFixed(1) : Math.round(Math.abs(v)).toString();
  return `${v < 0 ? "−" : "+"}${txt} / 1,000`;
};

export default function Scorer({ scorer, pol, price }) {
  const tech = useTech();
  const { features, feature_percentiles: fq, score_percentiles: sq, sample_users: sample } = scorer;
  const [values, setValues] = useState(sample[0]);
  const [res, setRes] = useState(null);
  const [presets, setPresets] = useState(null);
  const [error, setError] = useState(null);
  // which button is highlighted; null once the user moves a slider (a custom person)
  const [active, setActive] = useState(null);

  // Score the sample once to build presets at low / middle / high uplift.
  useEffect(() => {
    scoreUsers(sample)
      .then((scores) => {
        const order = scores.map((s, i) => [s.uplift, i]).sort((a, b) => a[0] - b[0]);
        const pick = (q) => sample[order[Math.min(order.length - 1, Math.floor(q * order.length))][1]];
        const p = { "Rarely responds": pick(0.1), Typical: pick(0.5), "Very responsive": pick(0.97) };
        setPresets(p);
        setValues(p.Typical);
        setActive("Typical");
      })
      .catch((e) => setError(String(e)));
  }, [sample]);

  useEffect(() => {
    let live = true;
    scoreUsers([values]).then(([r]) => live && setRes(r)).catch((e) => setError(String(e)));
    return () => { live = false; };
  }, [values]);

  const pct = res ? percentileOfValue(sq, res.uplift) : null;
  const top = pct === null ? null : Math.max(1, Math.round(100 - pct));
  const show = res && (pol.x === 1 || (pol.x === 0.5 && pct >= 50));
  const rule = pol.x === 1 ? "we show everyone" : pol.x === 0.5 ? "we show the top half" : "we show nobody";
  const featPct = useMemo(() => values.map((v, i) => percentileOfValue(fq[features[i]], v)), [values, fq, features]);

  return (
    <section className="card scorer" aria-label="Should this person see the ad">
      <div className="scorer-head">
        <div>
          <Swap as="div" className="lbl" s="Should this person see the ad?" t="Individual uplift · S-Learner (ONNX, in browser)" />
          <Swap as="div" className="one tight" s="Pick a person or move the sliders" t="τ(x) = P(visit | x, ad) − P(visit | x, no ad)" />
        </div>
        <div className="btns">
          <button className={`btn ${active === "Random person" ? "primary" : ""}`} aria-pressed={active === "Random person"}
            onClick={() => { setValues(sample[Math.floor(Math.random() * sample.length)]); setActive("Random person"); }}>
            Random person
          </button>
          {presets && Object.entries(presets).map(([k, v]) => (
            <button key={k} className={`btn ${active === k ? "primary" : ""}`} aria-pressed={active === k}
              onClick={() => { setValues(v); setActive(k); }}>
              {k}
            </button>
          ))}
        </div>
      </div>
      <div className="scorer-body">
        <div className="sliders">
          {features.map((f, i) => (
            <label key={f} className="slider">
              <span className="slider-head"><span>{f}</span><span className="mono">{tech ? values[i].toFixed(3) : `p${Math.round(featPct[i])}`}</span></span>
              <input type="range" min="0" max="100" step="1" value={Math.round(featPct[i])}
                onChange={(e) => { const next = values.slice(); next[i] = valueAtPercentile(fq[f], +e.target.value); setValues(next); setActive(null); }} />
            </label>
          ))}
          <Swap as="div" className="one" s="Traits are anonymized, so each slider is a percentile" t="Slider = feature percentile; value interpolated from the 10% sample" />
        </div>
        <div className="result">
          {error ? <div className="one">Model failed to load: {error}</div> : !res ? <div className="one">Loading model…</div> : (
            <>
              <Swap as="div" className="lbl" s="Extra visitors per 1,000 like them" t="Predicted uplift τ(x)" />
              <div className="big res-num" key={tech ? "t" : "s"}>
                {tech ? fmtPP(res.uplift, 3) : perThousand(res.uplift)}
              </div>
              <div className="one" key={`r${tech}`}>
                {tech ? `p(ad) ${res.pAd.toFixed(3)} · p(no ad) ${res.pNoAd.toFixed(3)} · percentile ${pct.toFixed(0)}` : `More responsive than ${Math.round(pct)}% of people`}
              </div>
              <div className="pct-track" aria-hidden="true">
                {pol.x === 0.5 && <span className="pct-cut" style={{ left: "50%" }} />}
                <span className="pct-pin" style={{ left: `${pct}%` }} />
              </div>
              <div className={`verdict ${show ? "good" : "bad"}`}>
                <span className="vdot" aria-hidden="true" />
                <span className="vword">{show ? "Show the ad" : "Skip this person"}</span>
              </div>
              <div className="one" key={`d${tech}`}>{tech ? `Rule at CPM ${fmtPrice(price)}: target share ${pol.x === null ? 0 : pol.x * 100}% · rank top ${top}%` : `At ${fmtPrice(price)} per ad, ${rule}`}</div>
            </>
          )}
        </div>
      </div>
    </section>
  );
}
