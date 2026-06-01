interface Props {
  speedKmh: number;
  maxKmh?: number;
}

export default function SpeedGauge({ speedKmh, maxKmh = 150 }: Props) {
  const clamp = Math.min(speedKmh, maxKmh);
  const pct   = clamp / maxKmh;

  // Arc: sweep from -135° to +135° (total 270°)
  const START_DEG  = -135;
  const SWEEP_DEG  = 270;
  const cx = 80, cy = 80, r = 62;

  function polarToXY(deg: number) {
    const rad = (deg * Math.PI) / 180;
    return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
  }

  function arcPath(startDeg: number, endDeg: number) {
    const s = polarToXY(startDeg);
    const e = polarToXY(endDeg);
    const large = endDeg - startDeg > 180 ? 1 : 0;
    return `M ${s.x} ${s.y} A ${r} ${r} 0 ${large} 1 ${e.x} ${e.y}`;
  }

  const trackEnd  = START_DEG + SWEEP_DEG;
  const needleEnd = START_DEG + pct * SWEEP_DEG;

  // Colour stops
  const color = speedKmh > maxKmh * 0.85 ? '#e8000d'
              : speedKmh > maxKmh * 0.6  ? '#ff6b00'
              : '#00d4ff';

  return (
    <svg viewBox="0 0 160 120" className="w-full h-full select-none" style={{ maxHeight: 120 }}>
      {/* Track */}
      <path d={arcPath(START_DEG, trackEnd)} fill="none" stroke="#1e1e1e" strokeWidth={8} strokeLinecap="round" />
      {/* Active arc */}
      {pct > 0 && (
        <path d={arcPath(START_DEG, needleEnd)} fill="none" stroke={color} strokeWidth={8}
          strokeLinecap="round"
          style={{ filter: `drop-shadow(0 0 4px ${color})` }} />
      )}
      {/* Tick marks */}
      {Array.from({ length: 11 }, (_, i) => {
        const deg = START_DEG + (i / 10) * SWEEP_DEG;
        const inner  = { x: cx + (r - 10) * Math.cos((deg * Math.PI) / 180), y: cy + (r - 10) * Math.sin((deg * Math.PI) / 180) };
        const outer  = { x: cx + (r + 2)  * Math.cos((deg * Math.PI) / 180), y: cy + (r + 2)  * Math.sin((deg * Math.PI) / 180) };
        return <line key={i} x1={inner.x} y1={inner.y} x2={outer.x} y2={outer.y} stroke="#333" strokeWidth={i % 5 === 0 ? 2 : 1} />;
      })}
      {/* Speed value */}
      <text x={cx} y={cy + 4} textAnchor="middle" dominantBaseline="middle"
        style={{ fontFamily: '"Courier New", monospace', fontWeight: 700, fontSize: 22, fill: '#fff' }}>
        {Math.round(speedKmh)}
      </text>
      <text x={cx} y={cy + 20} textAnchor="middle"
        style={{ fontFamily: 'Inter, sans-serif', fontSize: 10, fill: '#666', letterSpacing: '0.1em' }}>
        KM/H
      </text>
    </svg>
  );
}
