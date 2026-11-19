export default function Header({ tech, onToggle }) {
  return (
    <header className="nav">
      <div className="brand">
        <span className="logo" aria-hidden="true" />
        <b>Ad impact</b>
        <span className="muted">· Criteo campaign</span>
      </div>
      <button className="tswitch" role="switch" aria-checked={tech} onClick={onToggle}>
        Technical
        <span className="knob" aria-hidden="true" />
      </button>
    </header>
  );
}
