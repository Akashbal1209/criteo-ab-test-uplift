// Shows the plain-language text or the technical text, with a short fade when the mode changes.
import { useTech } from "../tech.js";

export default function Swap({ s, t, as: Tag = "span", className = "", ...rest }) {
  const tech = useTech();
  const text = tech ? t : s;
  return (
    <Tag key={tech ? "t" : "s"} className={`swap ${className}`} {...rest}>
      {text}
    </Tag>
  );
}
