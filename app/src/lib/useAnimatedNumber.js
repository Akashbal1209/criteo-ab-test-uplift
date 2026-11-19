import { useEffect, useRef, useState } from "react";

/**
 * Smoothly counts from the previous value to the new one (ease-out),
 * so changing the ad price visibly rolls the numbers instead of jumping.
 * A timer guarantees the final value is always shown, even if the browser
 * pauses animation frames (background tabs, hidden panes).
 */
export default function useAnimatedNumber(target, duration = 650) {
  const [value, setValue] = useState(target);
  const from = useRef(target);

  useEffect(() => {
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const finish = () => {
      from.current = target;
      setValue(target);
    };
    if (reduce || document.hidden || !Number.isFinite(target)) {
      finish();
      return undefined;
    }
    const start = performance.now();
    const a = from.current;
    let raf = 0;
    const step = (now) => {
      const p = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - p, 3);
      const v = a + (target - a) * eased;
      from.current = v;
      setValue(v);
      if (p < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    const done = setTimeout(() => { cancelAnimationFrame(raf); finish(); }, duration + 80);
    return () => { cancelAnimationFrame(raf); clearTimeout(done); };
  }, [target, duration]);

  return value;
}
