import { useEffect, useMemo, useState } from "react";
import { TechContext } from "./tech.js";
import Background from "./components/Background.jsx";
import Header from "./components/Header.jsx";
import Swap from "./components/Swap.jsx";
import Hero from "./components/Hero.jsx";
import Kpis from "./components/Kpis.jsx";
import TargetChart from "./components/TargetChart.jsx";
import Decision from "./components/Decision.jsx";
import Scorer from "./components/Scorer.jsx";
import { bestShare, fmtInt, fmtP, fmtPP, policy, profitAt } from "./lib/metrics.js";

export default function App() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [tech, setTech] = useState(false);
  const [price, setPrice] = useState(0.5);

  useEffect(() => {
    fetch("/data/app_data.json")
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then(setData)
      .catch((e) => setError(String(e)));
  }, []);

  const derived = useMemo(() => {
    if (!data) return null;
    const a = data.assumptions, curve = data.curves.s_learner;
    const pol = policy(price);
    return { pol, result: profitAt(curve, pol.x, price, a), best: bestShare(curve, price, a) };
  }, [data, price]);

  if (error) return <div className="center">Could not load results: {error}</div>;
  if (!data) return <div className="center">Loading results…</div>;

  const e = data.experiment, a = data.assumptions;
  return (
    <TechContext.Provider value={tech}>
      <Background />
      <main className="page">
        <Header tech={tech} onToggle={() => setTech((t) => !t)} />
        <div className="headline">
          <Swap as="h1" className="h1" s={`Ads brought ${Math.round(e.visit.itt * 1000)} extra visitors per 1,000 people.`}
            t={`ITT on visit: ${fmtPP(e.visit.itt)} (z = ${e.visit.z}, ${fmtP(e.visit.p)}).`} />
          <Swap as="p" className="h2" s={`Randomized test · ${(e.n * 10 / 1e6).toFixed(0)} million people in the full data · ${Math.round(e.treatment_share * 100)}% could see ads`}
            t={`n = ${fmtInt(e.n)} (10% sample) · treatment share ${(e.treatment_share * 100).toFixed(2)}% · SRM p = ${e.srm_p} · max |SMD| ${e.max_abs_smd}`} />
        </div>
        <Hero e={e} />
        <Kpis e={e} profit={derived.result.profit} price={price} />
        <div className="main">
          <TargetChart curves={data.curves} models={data.models} selX={derived.pol.x} />
          <Decision prices={a.price_options} price={price} setPrice={setPrice} pol={derived.pol}
            result={derived.result} best={derived.best} visitValue={a.visit_value} />
        </div>
        <Scorer scorer={data.scorer} pol={derived.pol} price={price} />
        <footer className="foot">
          <Swap s="Data: Criteo AI Lab uplift dataset. Ad prices and the ₹5 value of a visit are assumptions."
            t={`Criteo uplift v2.1, 10% random sample (seed 42). Prognostic imbalance z = ${e.prognostic_z}; adjusted lift ${fmtPP(e.visit.adjusted)}. Visit value ₹${a.visit_value}; profit per ${fmtInt(a.population)} users.`} />
        </footer>
      </main>
    </TechContext.Provider>
  );
}
