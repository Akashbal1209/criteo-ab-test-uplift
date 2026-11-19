import Swap from "./Swap.jsx";
import { fmtPrice } from "../lib/metrics.js";
import { useTech } from "../tech.js";

export default function Decision({ prices, price, setPrice, pol, result, best, visitValue }) {
  const tech = useTech();
  const x = pol.x;
  const roi = result.spend > 0 ? result.profit / result.spend : null;
  return (
    <section className="card" aria-label="Decision">
      <Swap as="div" className="lbl" s="Ad price" t="Cost per impression" />
      <Swap as="div" className="one tight" s="What we pay each time an ad is shown" t={`Visit value assumed ₹${visitValue}`} />
      <div className="chips" role="radiogroup" aria-label="Ad price">
        {prices.map((p) => (
          <button key={p} role="radio" aria-checked={p === price} className={`chip ${p === price ? "on" : ""}`} onClick={() => setPrice(p)}>
            {fmtPrice(p)}
          </button>
        ))}
      </div>
      <div className={`verdict ${pol.tone}`}>
        <span className="vdot" aria-hidden="true" />
        <span className="vword">{pol.word}</span>
      </div>
      <div className="one" key={tech ? "t" : "s"}>
        {tech
          ? `Break-even lift per impression ${((price / visitValue) * 100).toFixed(0)}% · profit-max share ${Math.round(best.x * 100)}%`
          : pol.line}
      </div>
      <div className="rows">
        <div>
          <Swap s="Show ads to" t="Targeted share" />
          <b>{x === null ? (tech ? "0%" : "Nobody") : x === 1 ? (tech ? "100%" : "Everyone") : tech ? "50%" : "Top half"}</b>
        </div>
        <div>
          <Swap s="Return per ₹1 spent" t="ROI" />
          <b>{roi === null ? "—" : tech ? `${Math.round(roi * 100)}%` : `₹${(1 + roi).toFixed(2)}`}</b>
        </div>
      </div>
    </section>
  );
}
