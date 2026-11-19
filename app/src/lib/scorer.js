// Runs the exported S-Learner (ONNX) in the browser.
// Uplift = P(visit | features, ad) − P(visit | features, no ad).
import * as ort from "onnxruntime-web/wasm";

// Full URL (with origin) on purpose: Vite's dev server rewrites root-relative dynamic imports,
// which breaks loading the runtime from /public. A full URL is left untouched.
ort.env.wasm.wasmPaths = `${window.location.origin}/ort/`;
ort.env.wasm.numThreads = 1; // no cross-origin isolation needed on a static host

let sessionPromise = null;
// onnxruntime-web does not allow two run() calls on one session at the same time,
// so every request waits for the previous one.
let queue = Promise.resolve();

function session() {
  if (!sessionPromise) sessionPromise = ort.InferenceSession.create("/model/s_learner.onnx");
  return sessionPromise;
}

/**
 * Score many users at once.
 * @param {number[][]} users  rows of 12 feature values (f0..f11)
 * @returns {Promise<{uplift:number, pAd:number, pNoAd:number}[]>}
 */
export function scoreUsers(users) {
  const job = queue.then(() => runBatch(users));
  queue = job.catch(() => {});
  return job;
}

async function runBatch(users) {
  const s = await session();
  const n = users.length, k = users[0].length + 1;
  const data = new Float32Array(2 * n * k);
  users.forEach((u, i) => {
    for (const [row, t] of [[i, 1], [n + i, 0]]) {
      data.set(u, row * k);
      data[row * k + k - 1] = t;
    }
  });
  const input = new ort.Tensor("float32", data, [2 * n, k]);
  const out = await s.run({ [s.inputNames[0]]: input });
  const probs = out.probabilities.data; // [2n, 2]
  return users.map((_, i) => {
    const pAd = probs[i * 2 + 1], pNoAd = probs[(n + i) * 2 + 1];
    return { uplift: pAd - pNoAd, pAd, pNoAd };
  });
}
