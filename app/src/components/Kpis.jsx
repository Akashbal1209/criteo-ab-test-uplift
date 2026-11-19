import Swap from "./Swap.jsx";
import { fmtPP, fmtPct, fmtRupeesK, fmtPrice } from "../lib/metrics.js";
import { useTech } from "../tech.js";
import useAnimatedNumber from "../lib/useAnimatedNumber.js";

function Kpi({ label, value, line, tone }) {
  return (
    <div className={`card kpi ${tone || ""}`}>
      <Swap as="div" className="lbl" {...label} />
      <Swap as="div" className="big" {...value} />
      <Swap as="div" className="one" {...line} />
    </div>
  );
}

export default function Kpis({ e, profit, price }) {
  const tech = useTech();
  const c = e.cace;
  const shownProfit = useAnimatedNumber(profit);
  return (
    <section className="k4" aria-label="Key numbers">
      <Kpi label={{ s: "Saw an ad", t: "Exposure rate" }}
        value={{ s: `${Math.round(e.exposure_rate * 1000)} / 1,000`, t: fmtPct(e.exposure_rate) }}
        line={{ s: "Most people picked never saw one", t: "P(exposed | treatment); 0% in control" }} />
      <Kpi label={{ s: "Effect if seen", t: "CACE (Wald / IV)" }}
        value={{ s: `+${Math.round(c.estimate * 100)} / 100`, t: fmtPP(c.estimate, 1) }}
        line={{ s: "Extra visitors among those who saw it", t: `95% CI [${(c.ci[0] * 100).toFixed(1)}, ${(c.ci[1] * 100).toFixed(1)}] · first-stage F ${Math.round(c.first_stage_f / 1000)}k` }} />
      <Kpi tone="warn" label={{ s: "Wrong method says", t: "Naive exposed vs not" }}
        value={{ s: `+${Math.round(e.naive * 100)} / 100`, t: fmtPP(e.naive, 1) }}
        line={{ s: "Too high: they were already active", t: `Selection bias = ${(e.selection_bias * 100).toFixed(1)} pp` }} />
      <div className="card kpi">
        <Swap as="div" className="lbl" s="Profit" t="Profit per 1M users" />
        <div className="big">{fmtRupeesK(shownProfit)}</div>
        <div className="one" key={tech ? "t" : "s"}>
          {tech ? `CPM ${fmtPrice(price)} · S-Learner ranking` : `Per 1M people, at ${fmtPrice(price)} per ad`}
        </div>
      </div>
    </section>
  );
}
