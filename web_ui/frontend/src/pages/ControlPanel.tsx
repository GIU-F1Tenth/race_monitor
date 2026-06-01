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
  bgDeep:    '#030C1C',
  bgCard:    '#060E1E',
  bgHover:   '#0A1628',
  border:    '#0D1E3A',
  borderBri: '#152D55',
  text:      '#D8E8FF',
  muted:     '#3A5580',
  faint:     '#162340',

  blue:      '#1A6FFF',
  blueBri:   '#4DA0FF',
  blueGlow:  '#1A6FFF',
  blueDeep:  '#0D3B99',

  green:     '#00D96A',
  gold:      '#FFB800',
  amber:     '#FF9500',
  red:       '#FF3D5A',
};

function statusColor(status: string): string {
  const s = status.toUpperCase();
  if (s.includes('RACING'))                    return T.green;
  if (s.includes('PAUSED'))                    return T.amber;
  if (s.includes('FINISHED') || s.includes('COMPLETE')) return T.blueBri;
  if (s.includes('CRASHED'))                   return T.red;
  return T.muted;
}

// ── StatusPip ─────────────────────────────────────────────────────────────────
function StatusPip({ live }: { live: boolean }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      <div style={{
        width: 8, height: 8, borderRadius: '50%',
        background: live ? T.green : T.red,
        boxShadow: live ? `0 0 8px ${T.green}` : `0 0 8px ${T.red}`,
        animation: live ? 'pip-pulse 1.8s ease-in-out infinite' : 'none',
      }} />
      <span style={{ fontFamily: "'Barlow Condensed'", fontWeight: 600, fontSize: 11, letterSpacing: '0.15em', color: live ? T.green : T.red }}>
        {live ? 'LIVE' : 'OFFLINE'}
      </span>
    </div>
  );
}

// ── InfoRow ───────────────────────────────────────────────────────────────────
function InfoRow({ label, value, valueColor, live }: {
  label: string; value: string; valueColor?: string; live?: boolean;
}) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '7px 0', borderBottom: `1px solid ${T.border}` }}>
      <span style={{ fontFamily: "'Barlow Condensed'", fontSize: 11, fontWeight: 600, letterSpacing: '0.12em', color: T.muted }}>
        {label}
      </span>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        {live && (
          <div style={{ width: 5, height: 5, borderRadius: '50%', background: T.green, boxShadow: `0 0 5px ${T.green}`, animation: 'pip-pulse 1s ease-in-out infinite', flexShrink: 0 }} />
        )}
        <span style={{ fontFamily: "'Barlow Condensed'", fontSize: 14, fontWeight: 700, letterSpacing: '0.05em', color: valueColor || T.text, maxWidth: 200, textAlign: 'right', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
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
      display: 'flex', alignItems: 'center', gap: 12,
      padding: '5px 10px',
      background: isBest ? 'rgba(255,184,0,0.07)' : isCurrent ? 'rgba(77,160,255,0.07)' : 'transparent',
      borderRadius: 4,
      borderLeft: `2px solid ${isBest ? T.gold : isCurrent ? T.blueBri : 'transparent'}`,
    }}>
      <span style={{ fontFamily: "'Barlow Condensed'", fontSize: 11, fontWeight: 600, letterSpacing: '0.1em', color: T.muted, minWidth: 24 }}>
        L{lap}
      </span>
      <span style={{ fontFamily: "'Orbitron', monospace", fontSize: 13, fontWeight: isBest ? 700 : 400, color: isBest ? T.gold : isCurrent ? T.blueBri : '#6080A8', letterSpacing: '0.05em' }}>
        {fmtSec(time)}
      </span>
      {isBest && (
        <span style={{ fontFamily: "'Barlow Condensed'", fontSize: 9, fontWeight: 700, color: T.gold, letterSpacing: '0.15em', marginLeft: 'auto' }}>
          BEST
        </span>
      )}
    </div>
  );
}

// ── CtrlBtn ───────────────────────────────────────────────────────────────────
const BTN_COLORS = {
  blue:  { bg: `${T.blueDeep}55`, border: T.blue,  text: T.blueBri, glow: T.blueGlow, hover: `${T.blueDeep}99` },
  amber: { bg: '#1C1000',          border: T.amber,  text: T.amber,   glow: T.amber,    hover: '#2A1800' },
  green: { bg: '#001A0C',          border: T.green,  text: T.green,   glow: T.green,    hover: '#00280F' },
  ghost: { bg: T.bgCard,           border: T.border, text: T.muted,   glow: 'transparent', hover: T.bgHover },
};

interface BtnProps {
  label: string; sub?: string; path: string;
  color: keyof typeof BTN_COLORS;
  onAction: (path: string, label: string) => Promise<void>;
  loading: string | null;
}

function CtrlBtn({ label, sub, path, color, onAction, loading }: BtnProps) {
  const [hovered, setHovered] = useState(false);
  const [pressed, setPressed] = useState(false);
  const c = BTN_COLORS[color];
  const isLoading = loading === label;

  const handleClick = async () => {
    setPressed(true);
    setTimeout(() => setPressed(false), 150);
    await onAction(path, label);
  };

  return (
    <button
      onClick={handleClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      disabled={isLoading}
      style={{
        flex: '1 1 0', minWidth: 0,
        padding: '14px 16px',
        background: hovered ? c.hover : c.bg,
        border: `1px solid ${hovered || pressed ? c.border : T.border}`,
        borderRadius: 3,
        cursor: isLoading ? 'not-allowed' : 'pointer',
        transition: 'all 0.12s ease',
        boxShadow: hovered && color !== 'ghost' ? `0 0 18px ${c.glow}33, inset 0 0 10px ${c.glow}0A` : 'none',
        transform: pressed ? 'scale(0.98)' : 'scale(1)',
        position: 'relative', overflow: 'hidden',
      }}
    >
      <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: 1, background: hovered && color !== 'ghost' ? c.border : 'transparent', transition: 'background 0.12s' }} />
      <div style={{ fontFamily: "'Barlow Condensed'", fontSize: 12, fontWeight: 700, letterSpacing: '0.18em', color: isLoading ? T.faint : c.text, textTransform: 'uppercase' }}>
        {isLoading ? '···' : label}
      </div>
      {sub && (
        <div style={{ fontFamily: "'Barlow Condensed'", fontSize: 10, fontWeight: 400, letterSpacing: '0.1em', color: T.faint, marginTop: 2 }}>
          {sub}
        </div>
      )}
    </button>
  );
}

// ── Main ──────────────────────────────────────────────────────────────────────
export default function ControlPanel() {
  const { state, wsConnected, lapElapsed } = useRaceData();
  const [loading, setLoading] = useState<string | null>(null);
  const [feedback, setFeedback] = useState<{ msg: string; ok: boolean } | null>(null);
  const feedbackTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const act = async (path: string, label: string) => {
    setLoading(label);
    const result = await callService(path);
    setLoading(null);
    if (feedbackTimer.current) clearTimeout(feedbackTimer.current);
    setFeedback({ msg: result.message || (result.success ? 'OK' : 'Failed'), ok: result.success });
    feedbackTimer.current = setTimeout(() => setFeedback(null), 3500);
  };

  useEffect(() => () => { if (feedbackTimer.current) clearTimeout(feedbackTimer.current); }, []);

  const bestLap = state.lap_times.length > 0 ? Math.min(...state.lap_times) : null;
  const raceStatus = state.race_status || 'WAITING';

  return (
    <div style={{
      minHeight: '100vh',
      background: T.bgDeep,
      backgroundImage: `radial-gradient(ellipse 80% 50% at 50% -10%, #0D2A6020, transparent)`,
      display: 'flex', flexDirection: 'column', alignItems: 'center',
      padding: '0 16px 40px',
      fontFamily: "'Barlow Condensed', sans-serif",
    }}>

      <style>{`
        @keyframes pip-pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.5;transform:scale(.75)} }
        @keyframes slide-in  { from{opacity:0;transform:translateY(-6px)} to{opacity:1;transform:translateY(0)} }
        @keyframes tick      { 0%,100%{opacity:1} 50%{opacity:.88} }
        * { box-sizing:border-box; margin:0; padding:0; }
        ::-webkit-scrollbar       { width:4px }
        ::-webkit-scrollbar-track { background:${T.bgDeep} }
        ::-webkit-scrollbar-thumb { background:${T.borderBri}; border-radius:2px }
      `}</style>

      {/* ── Header ── */}
      <header style={{
        width: '100%', maxWidth: 940,
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '22px 0 18px',
        borderBottom: `1px solid ${T.border}`,
      }}>
        {/* Logo + name */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 18 }}>
          <img
            src="/logo.png"
            alt="Celeritas"
            style={{ width: 80, height: 80, objectFit: 'contain', filter: 'drop-shadow(0 0 12px #1A6FFF55)' }}
          />
          <div>
            <div style={{ fontFamily: "'Rajdhani', sans-serif", fontSize: 30, fontWeight: 700, letterSpacing: '0.2em', color: T.text, lineHeight: 1 }}>
              CELERITAS
            </div>
            <div style={{ fontFamily: "'Barlow Condensed'", fontSize: 11, fontWeight: 600, letterSpacing: '0.25em', color: T.blueBri, marginTop: 4 }}>
              RACE CONTROL
            </div>
          </div>
        </div>

        {/* Status */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 20 }}>
          <div style={{ textAlign: 'right' }}>
            <div style={{ fontSize: 10, fontWeight: 600, letterSpacing: '0.15em', color: T.faint }}>ROS BRIDGE</div>
            <div style={{ fontSize: 11, fontWeight: 600, letterSpacing: '0.1em', color: state.race_monitor_connected ? T.green : T.muted, marginTop: 2 }}>
              {state.race_monitor_connected ? 'MONITOR ●' : 'NO MONITOR ○'}
            </div>
          </div>
          <div style={{ width: 1, height: 28, background: T.border }} />
          <StatusPip live={wsConnected} />
        </div>
      </header>

      {/* ── Blue accent stripe ── */}
      <div style={{
        width: '100%', maxWidth: 940, height: 2, marginBottom: 24,
        background: `linear-gradient(90deg, ${T.blueBri} 0%, ${T.blue} 35%, transparent 100%)`,
      }} />

      {/* ── Main cards ── */}
      <div style={{ width: '100%', maxWidth: 940, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 12 }}>

        {/* Lap timer */}
        <div style={{ background: T.bgCard, border: `1px solid ${T.border}`, borderRadius: 4, padding: '22px 26px', position: 'relative', overflow: 'hidden' }}>
          <div style={{ position: 'absolute', top: 0, left: 0, width: 3, height: '100%', background: `linear-gradient(180deg, ${T.blueBri}, ${T.blue})` }} />
          <div style={{ fontSize: 10, fontWeight: 600, letterSpacing: '0.2em', color: T.muted, marginBottom: 10 }}>CURRENT LAP</div>
          <div style={{
            fontFamily: "'Orbitron', monospace", fontSize: 46, fontWeight: 900,
            color: state.race_running ? T.text : T.faint,
            letterSpacing: '-0.02em', lineHeight: 1,
            animation: state.race_running ? 'tick 1s ease-in-out infinite' : 'none',
            transition: 'color 0.4s',
            textShadow: state.race_running ? `0 0 30px ${T.blue}55` : 'none',
          }}>
            {fmtMs(lapElapsed)}
          </div>
          <div style={{ display: 'flex', gap: 20, marginTop: 12 }}>
            <div>
              <div style={{ fontSize: 9, fontWeight: 600, letterSpacing: '0.15em', color: T.faint }}>LAP</div>
              <div style={{ fontFamily: "'Orbitron', monospace", fontSize: 22, fontWeight: 700, color: T.muted }}>
                {String(state.lap_count).padStart(2, '0')}
              </div>
            </div>
            {bestLap != null && (
              <div>
                <div style={{ fontSize: 9, fontWeight: 600, letterSpacing: '0.15em', color: T.faint }}>BEST</div>
                <div style={{ fontFamily: "'Orbitron', monospace", fontSize: 22, fontWeight: 700, color: T.gold, textShadow: `0 0 12px ${T.gold}66` }}>
                  {fmtSec(bestLap)}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Race data */}
        <div style={{ background: T.bgCard, border: `1px solid ${T.border}`, borderRadius: 4, padding: '22px 26px', position: 'relative', overflow: 'hidden' }}>
          <div style={{ position: 'absolute', top: 0, left: 0, width: 3, height: '100%', background: statusColor(raceStatus) }} />
          <div style={{ fontSize: 10, fontWeight: 600, letterSpacing: '0.2em', color: T.muted, marginBottom: 10 }}>RACE DATA</div>
          <InfoRow label="STATUS"     value={raceStatus}                        valueColor={statusColor(raceStatus)} />
          <InfoRow label="CONTROLLER" value={state.controller_name || '—'} />
          <InfoRow label="SPEED"      value={`${state.velocity.toFixed(2)} m/s`} live={state.velocity > 0} />
          <InfoRow label="POSITION"   value={state.position ? `[${state.position.x.toFixed(2)}, ${state.position.y.toFixed(2)}]` : '—'} />
        </div>
      </div>

      {/* ── Lap history ── */}
      <div style={{ width: '100%', maxWidth: 940, background: T.bgCard, border: `1px solid ${T.border}`, borderRadius: 4, padding: '16px 22px', marginBottom: 12 }}>
        <div style={{ fontSize: 10, fontWeight: 600, letterSpacing: '0.2em', color: T.muted, marginBottom: 10 }}>LAP HISTORY</div>
        {state.lap_times.length === 0 ? (
          <div style={{ fontSize: 13, color: T.faint, letterSpacing: '0.1em', padding: '4px 0' }}>NO LAPS RECORDED YET</div>
        ) : (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(160px, 1fr))', gap: 4, maxHeight: 180, overflowY: 'auto' }}>
            {state.lap_times.map((t, i) => (
              <LapEntry key={i} lap={i + 1} time={t} isBest={t === bestLap} isCurrent={i === state.lap_times.length - 1 && state.race_running} />
            ))}
          </div>
        )}
      </div>

      {/* ── Controls ── */}
      <div style={{ width: '100%', maxWidth: 940, background: T.bgCard, border: `1px solid ${T.border}`, borderRadius: 4, padding: '16px 22px' }}>
        <div style={{ fontSize: 10, fontWeight: 600, letterSpacing: '0.2em', color: T.muted, marginBottom: 12 }}>RACE CONTROL</div>
        <div style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
          <CtrlBtn label="PAUSE"     sub="[P]"           path="/api/race/pause"          color="amber" onAction={act} loading={loading} />
          <CtrlBtn label="RESUME"    sub="[R]"           path="/api/race/resume"         color="green" onAction={act} loading={loading} />
          <CtrlBtn label="RESET LAP" sub="[T]"           path="/api/race/reset_lap"      color="blue"  onAction={act} loading={loading} />
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <CtrlBtn label="IGNORE / RESET" sub="[I] · no save"      path="/api/race/reset"         color="ghost" onAction={act} loading={loading} />
          <CtrlBtn label="END + SAVE"     sub="[F] · completes race" path="/api/race/force_complete" color="blue"  onAction={act} loading={loading} />
        </div>
      </div>

      {/* ── Feedback toast ── */}
      {feedback && (
        <div style={{
          position: 'fixed', bottom: 24, right: 24,
          background: feedback.ok ? '#001A10' : '#12000A',
          border: `1px solid ${feedback.ok ? T.green : T.red}`,
          borderRadius: 3, padding: '10px 16px',
          fontFamily: "'Barlow Condensed'",
          fontSize: 13, fontWeight: 600, letterSpacing: '0.08em',
          color: feedback.ok ? T.green : T.red,
          animation: 'slide-in 0.2s ease',
          zIndex: 1000, maxWidth: 280,
        }}>
          {feedback.ok ? '✓ ' : '✗ '}{feedback.msg}
        </div>
      )}
    </div>
  );
}
