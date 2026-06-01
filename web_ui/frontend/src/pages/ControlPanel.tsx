import { useState, useRef, useEffect } from 'react';
import { useRaceData, fmtMs, fmtSec } from '../hooks/useRaceData';

const API = `http://${window.location.hostname}:8082`;

async function callService(path: string): Promise<{ success: boolean; message: string }> {
  try {
    const res = await fetch(API + path, { method: 'POST' });
    return await res.json();
  } catch (e) {
    return { success: false, message: String(e) };
  }
}

// ── Theme ─────────────────────────────────────────────────────────────────────
const T = {
  bgDeep:  '#030C1C',
  bgCard:  '#060E1E',
  bgHover: '#0B1828',
  border:  '#0F2040',
  borderB: '#1A3560',

  // Text hierarchy — all clearly readable
  text:    '#E8F2FF',   // primary values
  label:   '#7AAAD8',   // row labels & section headers
  sub:     '#4A7099',   // key hints & secondary sub-text
  dim:     '#2A4A6A',   // true dividers / decorative only

  blue:    '#1A6FFF',
  blueBri: '#4DA0FF',
  blueD:   '#0A3080',

  green:   '#00E272',
  gold:    '#FFB800',
  amber:   '#FF9500',
  red:     '#FF2D4A',
};

function statusColor(s: string): string {
  const u = s.toUpperCase();
  if (u.includes('RACING'))                          return T.green;
  if (u.includes('PAUSED'))                          return T.amber;
  if (u.includes('FINISHED') || u.includes('COMPLETE')) return T.blueBri;
  if (u.includes('CRASHED'))                         return T.red;
  return T.sub;
}

// ── StatusPip — reflects actual race_monitor connectivity, not just WS ────────
function StatusPip({ wsConnected, monitorConnected }: { wsConnected: boolean; monitorConnected: boolean }) {
  const color = !wsConnected ? T.red : !monitorConnected ? T.amber : T.green;
  const label = !wsConnected ? 'OFFLINE' : !monitorConnected ? 'WAITING' : 'LIVE';
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      <div style={{
        width: 8, height: 8, borderRadius: '50%',
        background: color, boxShadow: `0 0 8px ${color}`,
        animation: wsConnected ? 'pip 1.8s ease-in-out infinite' : 'none',
      }} />
      <span style={{ fontFamily: "'Barlow Condensed'", fontWeight: 700, fontSize: 12, letterSpacing: '0.15em', color }}>
        {label}
      </span>
    </div>
  );
}

// ── InfoRow ───────────────────────────────────────────────────────────────────
function InfoRow({ label, value, valueColor, live }: {
  label: string; value: string; valueColor?: string; live?: boolean;
}) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '8px 0', borderBottom: `1px solid ${T.border}` }}>
      <span style={{ fontFamily: "'Barlow Condensed'", fontSize: 12, fontWeight: 600, letterSpacing: '0.14em', color: T.label }}>
        {label}
      </span>
      <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
        {live && (
          <div style={{ width: 5, height: 5, borderRadius: '50%', background: T.green, boxShadow: `0 0 6px ${T.green}`, animation: 'pip 1s ease-in-out infinite', flexShrink: 0 }} />
        )}
        <span style={{ fontFamily: "'Barlow Condensed'", fontSize: 15, fontWeight: 700, letterSpacing: '0.04em', color: valueColor || T.text, maxWidth: 200, textAlign: 'right', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {value || '—'}
        </span>
      </div>
    </div>
  );
}

// ── LapEntry ──────────────────────────────────────────────────────────────────
function LapEntry({ lap, time, isBest, isCurrent }: {
  lap: number; time: number; isBest: boolean; isCurrent: boolean;
}) {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 12, padding: '5px 10px',
      background: isBest ? 'rgba(255,184,0,0.08)' : isCurrent ? 'rgba(77,160,255,0.07)' : 'transparent',
      borderRadius: 4,
      borderLeft: `2px solid ${isBest ? T.gold : isCurrent ? T.blueBri : 'transparent'}`,
    }}>
      <span style={{ fontFamily: "'Barlow Condensed'", fontSize: 12, fontWeight: 600, letterSpacing: '0.1em', color: T.label, minWidth: 26 }}>
        L{lap}
      </span>
      <span style={{ fontFamily: "'Orbitron', monospace", fontSize: 13, fontWeight: isBest ? 700 : 400, color: isBest ? T.gold : isCurrent ? T.blueBri : T.sub, letterSpacing: '0.04em' }}>
        {fmtSec(time)}
      </span>
      {isBest && (
        <span style={{ fontFamily: "'Barlow Condensed'", fontSize: 9, fontWeight: 700, color: T.gold, letterSpacing: '0.15em', marginLeft: 'auto' }}>★ BEST</span>
      )}
    </div>
  );
}

// ── CtrlBtn ───────────────────────────────────────────────────────────────────
const BTNS = {
  red:   { bg: '#1F0008', border: T.red,    text: '#FF6070', glow: T.red,    hov: '#2E000D' },
  green: { bg: '#001A0D', border: T.green,  text: '#00FF88', glow: T.green,  hov: '#002A16' },
  amber: { bg: '#1A0E00', border: T.amber,  text: '#FFB040', glow: T.amber,  hov: '#2A1800' },
  blue:  { bg: '#050E2A', border: T.blue,   text: T.blueBri, glow: T.blue,   hov: '#0A1840' },
  ghost: { bg: T.bgCard,  border: T.border, text: T.label,   glow: 'none',   hov: T.bgHover },
};

interface BtnProps {
  label: string; sub?: string; path: string;
  color: keyof typeof BTNS;
  onAction: (path: string, label: string) => Promise<void>;
  loading: string | null;
}

function CtrlBtn({ label, sub, path, color, onAction, loading }: BtnProps) {
  const [hov, setHov] = useState(false);
  const [pressed, setPressed] = useState(false);
  const c = BTNS[color];
  const busy = loading === label;

  const click = async () => {
    setPressed(true);
    setTimeout(() => setPressed(false), 130);
    await onAction(path, label);
  };

  return (
    <button
      onClick={click}
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      disabled={busy}
      style={{
        flex: '1 1 0', minWidth: 0, padding: '15px 16px',
        background: hov ? c.hov : c.bg,
        border: `1px solid ${hov || pressed ? c.border : T.border}`,
        borderRadius: 3,
        cursor: busy ? 'not-allowed' : 'pointer',
        transition: 'all 0.11s ease',
        boxShadow: hov && color !== 'ghost' ? `0 0 20px ${c.glow}40, inset 0 0 12px ${c.glow}10` : 'none',
        transform: pressed ? 'scale(0.975)' : 'scale(1)',
        position: 'relative', overflow: 'hidden',
      }}
    >
      {/* top shimmer on hover */}
      <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: 1, background: hov && color !== 'ghost' ? c.border : 'transparent', transition: 'background 0.11s' }} />
      <div style={{ fontFamily: "'Barlow Condensed'", fontSize: 13, fontWeight: 800, letterSpacing: '0.2em', color: busy ? T.dim : c.text, textTransform: 'uppercase' }}>
        {busy ? '···' : label}
      </div>
      {sub && (
        <div style={{ fontFamily: "'Barlow Condensed'", fontSize: 11, fontWeight: 500, letterSpacing: '0.1em', color: T.sub, marginTop: 3 }}>
          {sub}
        </div>
      )}
    </button>
  );
}

// ── SectionLabel ──────────────────────────────────────────────────────────────
function SectionLabel({ text }: { text: string }) {
  return (
    <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: '0.22em', color: T.label, marginBottom: 12, textTransform: 'uppercase' }}>
      {text}
    </div>
  );
}

// ── Main ──────────────────────────────────────────────────────────────────────
export default function ControlPanel() {
  const { state, wsConnected, lapElapsed } = useRaceData();
  const [loading, setLoading] = useState<string | null>(null);
  const [feedback, setFeedback] = useState<{ msg: string; ok: boolean } | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const act = async (path: string, label: string) => {
    setLoading(label);
    const r = await callService(path);
    setLoading(null);
    if (timer.current) clearTimeout(timer.current);
    setFeedback({ msg: r.message || (r.success ? 'OK' : 'Failed'), ok: r.success });
    timer.current = setTimeout(() => setFeedback(null), 3500);
  };

  useEffect(() => () => { if (timer.current) clearTimeout(timer.current); }, []);

  const bestLap = state.lap_times.length > 0 ? Math.min(...state.lap_times) : null;
  // Strip the mode suffix e.g. "RACING-MANUAL" → "RACING"
  const raceStatus = (state.race_status || 'WAITING').split('-')[0].toUpperCase();
  const racing = state.race_running;

  return (
    <div style={{
      minHeight: '100vh', background: T.bgDeep,
      backgroundImage: `radial-gradient(ellipse 90% 40% at 50% -5%, #0D2A6025, transparent)`,
      display: 'flex', flexDirection: 'column', alignItems: 'center',
      padding: '0 16px 40px', fontFamily: "'Barlow Condensed', sans-serif",
    }}>
      <style>{`
        @keyframes pip  { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.45;transform:scale(.7)} }
        @keyframes fadein { from{opacity:0;transform:translateY(-5px)} to{opacity:1;transform:translateY(0)} }
        @keyframes tick { 0%,100%{opacity:1} 50%{opacity:.85} }
        * { box-sizing:border-box; margin:0; padding:0; }
        ::-webkit-scrollbar{width:4px}
        ::-webkit-scrollbar-track{background:${T.bgDeep}}
        ::-webkit-scrollbar-thumb{background:${T.borderB};border-radius:2px}
      `}</style>

      {/* ── Header ── */}
      <header style={{ width: '100%', maxWidth: 960, display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '20px 0 16px', borderBottom: `1px solid ${T.border}` }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 20 }}>
          <img src="/logo.png" alt="Celeritas" style={{ width: 96, height: 96, objectFit: 'contain', filter: `drop-shadow(0 0 14px ${T.blue}66)` }} />
          <div>
            <div style={{ fontFamily: "'Rajdhani', sans-serif", fontSize: 32, fontWeight: 700, letterSpacing: '0.22em', color: T.text, lineHeight: 1 }}>
              CELERITAS
            </div>
            <div style={{ fontFamily: "'Barlow Condensed'", fontSize: 12, fontWeight: 700, letterSpacing: '0.28em', color: T.blueBri, marginTop: 5 }}>
              RACE CONTROL
            </div>
          </div>
        </div>

        {/* ROS status */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 20 }}>
          <div style={{ textAlign: 'right' }}>
            <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: '0.18em', color: T.label }}>ROS BRIDGE</div>
            <div style={{ fontSize: 12, fontWeight: 700, letterSpacing: '0.1em', color: state.race_monitor_connected ? T.green : '#8090A8', marginTop: 3 }}>
              {state.race_monitor_connected ? 'MONITOR ●' : 'NO MONITOR ○'}
            </div>
          </div>
          <div style={{ width: 1, height: 32, background: T.border }} />
          <StatusPip wsConnected={wsConnected} monitorConnected={state.race_monitor_connected} />
        </div>
      </header>

      {/* blue stripe */}
      <div style={{ width: '100%', maxWidth: 960, height: 2, marginBottom: 22, background: `linear-gradient(90deg, ${T.blueBri}, ${T.blue} 40%, transparent)` }} />

      {/* ── Data cards ── */}
      <div style={{ width: '100%', maxWidth: 960, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 12 }}>

        {/* Timer */}
        <div style={{ background: T.bgCard, border: `1px solid ${T.border}`, borderRadius: 4, padding: '22px 26px', position: 'relative', overflow: 'hidden' }}>
          <div style={{ position: 'absolute', top: 0, left: 0, width: 3, height: '100%', background: `linear-gradient(180deg, ${T.blueBri}, ${T.blue})` }} />
          <SectionLabel text="Current Lap" />
          <div style={{
            fontFamily: "'Orbitron', monospace", fontSize: 48, fontWeight: 900,
            color: racing ? T.text : T.dim, letterSpacing: '-0.02em', lineHeight: 1,
            animation: racing ? 'tick 1s ease-in-out infinite' : 'none',
            transition: 'color 0.4s',
            textShadow: racing ? `0 0 32px ${T.blue}60` : 'none',
          }}>
            {fmtMs(lapElapsed)}
          </div>
          <div style={{ display: 'flex', gap: 24, marginTop: 14 }}>
            <div>
              <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.18em', color: T.sub }}>LAP</div>
              <div style={{ fontFamily: "'Orbitron', monospace", fontSize: 24, fontWeight: 700, color: T.label }}>
                {String(state.lap_count).padStart(2, '0')}
              </div>
            </div>
            {bestLap != null && (
              <div>
                <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.18em', color: T.sub }}>BEST</div>
                <div style={{ fontFamily: "'Orbitron', monospace", fontSize: 24, fontWeight: 700, color: T.gold, textShadow: `0 0 14px ${T.gold}80` }}>
                  {fmtSec(bestLap)}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Race data */}
        <div style={{ background: T.bgCard, border: `1px solid ${T.border}`, borderRadius: 4, padding: '22px 26px', position: 'relative', overflow: 'hidden' }}>
          <div style={{ position: 'absolute', top: 0, left: 0, width: 3, height: '100%', background: statusColor(raceStatus) }} />
          <SectionLabel text="Race Data" />
          <InfoRow label="STATUS"     value={raceStatus}  valueColor={statusColor(raceStatus)} />
          <InfoRow label="CONTROLLER" value={state.controller_name || '—'} />
          <InfoRow label="SPEED"      value={`${state.velocity.toFixed(2)} m/s`} live={state.velocity > 0} />
          <InfoRow label="POSITION"   value={state.position ? `[${state.position.x.toFixed(2)}, ${state.position.y.toFixed(2)}]` : '—'} />
        </div>
      </div>

      {/* ── Lap history ── */}
      <div style={{ width: '100%', maxWidth: 960, background: T.bgCard, border: `1px solid ${T.border}`, borderRadius: 4, padding: '16px 22px', marginBottom: 12 }}>
        <SectionLabel text="Lap History" />
        {state.lap_times.length === 0 ? (
          <div style={{ fontSize: 13, color: T.sub, letterSpacing: '0.1em' }}>NO LAPS RECORDED YET</div>
        ) : (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(165px, 1fr))', gap: 4, maxHeight: 180, overflowY: 'auto' }}>
            {state.lap_times.map((t, i) => (
              <LapEntry key={i} lap={i + 1} time={t} isBest={t === bestLap} isCurrent={i === state.lap_times.length - 1 && racing} />
            ))}
          </div>
        )}
      </div>

      {/* ── Controls ── */}
      <div style={{ width: '100%', maxWidth: 960, background: T.bgCard, border: `1px solid ${T.border}`, borderRadius: 4, padding: '16px 22px' }}>
        <SectionLabel text="Race Control" />
        <div style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
          <CtrlBtn label="PAUSE"     sub="[P]" path="/api/race/pause"     color="amber" onAction={act} loading={loading} />
          <CtrlBtn label="RESUME"    sub="[R]" path="/api/race/resume"    color="blue"  onAction={act} loading={loading} />
          <CtrlBtn label="RESET LAP" sub="[T]" path="/api/race/reset_lap" color="ghost" onAction={act} loading={loading} />
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <CtrlBtn label="IGNORE / RESET" sub="[I] · no save"       path="/api/race/reset"         color="red"   onAction={act} loading={loading} />
          <CtrlBtn label="END + SAVE"     sub="[F] · complete race"  path="/api/race/force_complete" color="green" onAction={act} loading={loading} />
        </div>
      </div>

      {/* ── Toast ── */}
      {feedback && (
        <div style={{
          position: 'fixed', bottom: 24, right: 24,
          background: feedback.ok ? '#001A0D' : '#1A000A',
          border: `1px solid ${feedback.ok ? T.green : T.red}`,
          borderRadius: 3, padding: '10px 16px',
          fontFamily: "'Barlow Condensed'", fontSize: 13, fontWeight: 700, letterSpacing: '0.08em',
          color: feedback.ok ? T.green : T.red,
          animation: 'fadein 0.18s ease', zIndex: 1000, maxWidth: 280,
        }}>
          {feedback.ok ? '✓ ' : '✗ '}{feedback.msg}
        </div>
      )}
    </div>
  );
}
