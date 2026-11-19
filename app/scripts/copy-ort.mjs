// Copy the onnxruntime-web WebAssembly files into public/ort so the app serves them
// itself (no CDN needed at runtime, works the same on Render).
import { cpSync, mkdirSync, readdirSync } from "node:fs";
import { join } from "node:path";

const src = join("node_modules", "onnxruntime-web", "dist");
const dst = join("public", "ort");
mkdirSync(dst, { recursive: true });
// Only the plain WebAssembly build is needed (the "jsep" WebGPU build is ~21 MB and unused).
const files = readdirSync(src).filter((f) => /^ort-wasm-simd-threaded\.(wasm|mjs)$/.test(f));
for (const f of files) cpSync(join(src, f), join(dst, f));
console.log(`Copied ${files.length} onnxruntime-web files to ${dst}`);
