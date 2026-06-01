import { useState, useRef, useEffect } from 'react';
import { useRaceData, fmtMs, fmtSec } from '../hooks/useRaceData';

// ── API ───────────────────────────────────────────────────────────────────────

const API = `http://${window.location.hostname}:8082`;

async function callService(path: string): Promise<{ success: boolean; message: string }> {
  try {
    const res = await fetch(API + path, { method: 'POST' });
    return await res.json();
  } catch (e) {
    return { success: false, message: String(e) };
  }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function statusColor(status: string): string {
  const s = status.toUpperCase();
  if (s.includes('RACING')) return '#00D96A';
  if (s.includes('PAUSED')) return '#FFB800';
  if (s.includes('FINISHED') || s.includes('COMPLETE')) return '#0099FF';
  if (s.includes('CRASHED')) return '#E8002D';
  return '#6B6B90';
}

// ── Sub-components ────────────────────────────────────────────────────────────

function StatusPip({ live }: { live: boolean }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      <div style={{
        width: 8, height: 8, borderRadius: '50%',
        background: live ? '#00D96A' : '#E8002D',
        boxShadow: live ? '0 0 8px #00D96A' : '0 0 8px #E8002D',
        animation: live ? 'pip-pulse 1.8s ease-in-out infinite' : 'none',
      }} />
      <span style={{
        fontFamily: "'Barlow Condensed', sans-serif",
        fontWeight: 600, fontSize: 11, letterSpacing: '0.15em',
        color: live ? '#00D96A' : '#E8002D',
      }}>
        {live ? 'LIVE' : 'OFFLINE'}
      </span>
    </div>
  );
}

function InfoRow({ label, value, valueColor, live }: {
  label: string; value: string; valueColor?: string; live?: boolean;
}) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '7px 0', borderBottom: '1px solid #1A1A2E' }}>
      <span style={{ fontFamily: "'Barlow Condensed', sans-serif", fontSize: 11, fontWeight: 600, letterSpacing: '0.12em', color: '#4A4A6A' }}>
        {label}
      </span>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        {live && (
          <div style={{ width: 5, height: 5, borderRadius: '50%', background: '#00D96A', boxShadow: '0 0 5px #00D96A', animation: 'pip-pulse 1s ease-in-out infinite', flexShrink: 0 }} />
        )}
        <span style={{ fontFamily: "'Barlow Condensed', sans-serif", fontSize: 14, fontWeight: 700, letterSpacing: '0.05em', color: valueColor || '#D0D0F0', maxWidth: 200, textAlign: 'right', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {value || '—'}
        </span>
      </div>
    </div>
  );
}

function LapEntry({ lap, time, isBest, isCurrent }: {
  lap: number; time: number; isBest: boolean; isCurrent: boolean;
}) {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 12,
      padding: '5px 10px',
      background: isBest ? 'rgba(255,184,0,0.08)' : isCurrent ? 'rgba(0,153,255,0.06)' : 'transparent',
      borderRadius: 4,
      borderLeft: `2px solid ${isBest ? '#FFB800' : isCurrent ? '#0099FF' : 'transparent'}`,
      transition: 'background 0.2s',
    }}>
      <span style={{
        fontFamily: "'Barlow Condensed', sans-serif", fontSize: 11, fontWeight: 600,
        letterSpacing: '0.1em', color: '#4A4A6A', minWidth: 24,
      }}>
        L{lap}
      </span>
      <span style={{
        fontFamily: "'Orbitron', monospace", fontSize: 13, fontWeight: isBest ? 700 : 400,
        color: isBest ? '#FFB800' : isCurrent ? '#0099FF' : '#9090B8',
        letterSpacing: '0.05em',
      }}>
        {fmtSec(time)}
      </span>
      {isBest && (
        <span style={{ fontFamily: "'Barlow Condensed', sans-serif", fontSize: 9, fontWeight: 700, color: '#FFB800', letterSpacing: '0.15em', marginLeft: 'auto' }}>
          BEST
        </span>
      )}
    </div>
  );
}

interface BtnProps {
  label: string;
  sub?: string;
  path: string;
  color: 'red' | 'amber' | 'green' | 'blue' | 'ghost';
  wide?: boolean;
  onAction: (path: string, label: string) => Promise<void>;
  loading: string | null;
}

const BTN_COLORS = {
  red:   { bg: '#1A0005', border: '#E8002D', text: '#FF3D5A', glow: '#E8002D', hover: '#2A0009' },
  amber: { bg: '#1A1000', border: '#FFB800', text: '#FFB800', glow: '#FFB800', hover: '#2A1A00' },
  green: { bg: '#001A0A', border: '#00D96A', text: '#00D96A', glow: '#00D96A', hover: '#002A0F' },
  blue:  { bg: '#00091A', border: '#0099FF', text: '#0099FF', glow: '#0099FF', hover: '#000F2A' },
  ghost: { bg: '#0F0F1C', border: '#2A2A45', text: '#8080A8', glow: 'transparent', hover: '#15152A' },
};

function CtrlBtn({ label, sub, path, color, wide, onAction, loading }: BtnProps) {
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
        flex: wide ? '1 1 100%' : '1 1 0',
        minWidth: 0,
        padding: '14px 16px',
        background: hovered ? c.hover : c.bg,
        border: `1px solid ${hovered || pressed ? c.border : '#1E1E35'}`,
        borderRadius: 3,
        cursor: isLoading ? 'not-allowed' : 'pointer',
        transition: 'all 0.12s ease',
        boxShadow: hovered && color !== 'ghost' ? `0 0 16px ${c.glow}22, inset 0 0 8px ${c.glow}08` : 'none',
        transform: pressed ? 'scale(0.98)' : 'scale(1)',
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      {/* top accent line */}
      {color !== 'ghost' && (
        <div style={{
          position: 'absolute', top: 0, left: 0, right: 0, height: 1,
          background: hovered ? c.border : 'transparent',
          transition: 'background 0.12s',
        }} />
      )}
      <div style={{ fontFamily: "'Barlow Condensed', sans-serif", fontSize: 12, fontWeight: 700, letterSpacing: '0.18em', color: isLoading ? '#4A4A6A' : c.text, textTransform: 'uppercase' }}>
        {isLoading ? '...' : label}
      </div>
      {sub && (
        <div style={{ fontFamily: "'Barlow Condensed', sans-serif", fontSize: 10, fontWeight: 400, letterSpacing: '0.1em', color: '#3A3A5A', marginTop: 2 }}>
          {sub}
        </div>
      )}
    </button>
  );
}

// ── Logo placeholder ──────────────────────────────────────────────────────────

function TeamLogo() {
  return (
    <div style={{ width: 42, height: 42, position: 'relative' }}>
      {/* Replace this div with <img src="/logo.png" ... /> when logo is available */}
      <svg viewBox="0 0 42 42" fill="none" xmlns="http://www.w3.org/2000/svg" style={{ width: '100%', height: '100%' }}>
        <circle cx="21" cy="21" r="20" stroke="#E8002D" strokeWidth="1.5" fill="none" />
        <circle cx="21" cy="21" r="14" stroke="#E8002D" strokeWidth="0.5" strokeDasharray="2 3" fill="none" opacity="0.4" />
        <text x="21" y="26" textAnchor="middle" fill="#E8002D"
          style={{ fontFamily: "'Rajdhani', sans-serif", fontSize: 18, fontWeight: 700, letterSpacing: '-1px' }}>
          C
        </text>
      </svg>
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

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
      background: '#07070D',
      backgroundImage: `
        linear-gradient(45deg, rgba(255,255,255,0.012) 25%, transparent 25%),
        linear-gradient(-45deg, rgba(255,255,255,0.012) 25%, transparent 25%),
        linear-gradient(45deg, transparent 75%, rgba(255,255,255,0.012) 75%),
        linear-gradient(-45deg, transparent 75%, rgba(255,255,255,0.012) 75%)
      `,
      backgroundSize: '4px 4px',
      backgroundPosition: '0 0, 0 2px, 2px -2px, -2px 0px',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      padding: '0 16px 32px',
      fontFamily: "'Barlow Condensed', sans-serif",
    }}>

      {/* ── Keyframes ── */}
      <style>{`
        @keyframes pip-pulse {
          0%, 100% { opacity: 1; transform: scale(1); }
          50% { opacity: 0.5; transform: scale(0.75); }
        }
        @keyframes slide-in {
          from { opacity: 0; transform: translateY(-6px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes timer-tick {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.92; }
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: #0C0C18; }
        ::-webkit-scrollbar-thumb { background: #2A2A45; border-radius: 2px; }
      `}</style>

      {/* ── Header ── */}
      <header style={{
        width: '100%', maxWidth: 900,
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '20px 0 16px',
        borderBottom: '1px solid #1A1A2E',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
          <TeamLogo />
          <div>
            <div style={{
              fontFamily: "'Rajdhani', sans-serif",
              fontSize: 22, fontWeight: 700, letterSpacing: '0.22em',
              color: '#F0F0FF', lineHeight: 1,
            }}>
              CELERITAS
            </div>
            <div style={{
              fontFamily: "'Barlow Condensed', sans-serif",
              fontSize: 10, fontWeight: 600, letterSpacing: '0.2em',
              color: '#E8002D', marginTop: 3,
            }}>
              RACE CONTROL
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 20 }}>
          <div style={{ textAlign: 'right' }}>
            <div style={{ fontSize: 10, fontWeight: 600, letterSpacing: '0.15em', color: '#3A3A5A' }}>ROS BRIDGE</div>
            <div style={{ fontSize: 11, fontWeight: 600, letterSpacing: '0.1em', color: state.race_monitor_connected ? '#00D96A' : '#E8002D', marginTop: 2 }}>
              {state.race_monitor_connected ? 'MONITOR ●' : 'NO MONITOR ○'}
            </div>
          </div>
          <div style={{ width: 1, height: 28, background: '#1A1A2E' }} />
          <StatusPip live={wsConnected} />
        </div>
      </header>

      {/* ── Red accent stripe ── */}
      <div style={{
        width: '100%', maxWidth: 900, height: 2,
        background: 'linear-gradient(90deg, #E8002D 0%, #E8002D 40%, transparent 100%)',
        marginBottom: 24,
      }} />

      {/* ── Main grid ── */}
      <div style={{
        width: '100%', maxWidth: 900,
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gap: 12,
        marginBottom: 12,
      }}>

        {/* Timer card */}
        <div style={{
          background: '#0C0C18',
          border: '1px solid #1A1A2E',
          borderRadius: 4,
          padding: '20px 24px',
          position: 'relative',
          overflow: 'hidden',
        }}>
          <div style={{ position: 'absolute', top: 0, left: 0, width: 3, height: '100%', background: '#E8002D' }} />
          <div style={{ fontSize: 10, fontWeight: 600, letterSpacing: '0.2em', color: '#4A4A6A', marginBottom: 8 }}>
            CURRENT LAP
          </div>
          <div style={{
            fontFamily: "'Orbitron', monospace",
            fontSize: 44, fontWeight: 900,
            color: state.race_running ? '#F0F0FF' : '#3A3A5A',
            letterSpacing: '-0.02em', lineHeight: 1,
            animation: state.race_running ? 'timer-tick 1s ease-in-out infinite' : 'none',
            transition: 'color 0.3s',
          }}>
            {fmtMs(lapElapsed)}
          </div>
          <div style={{ display: 'flex', gap: 16, marginTop: 10 }}>
            <div>
              <div style={{ fontSize: 9, fontWeight: 600, letterSpacing: '0.15em', color: '#3A3A5A' }}>LAP</div>
              <div style={{ fontFamily: "'Orbitron', monospace", fontSize: 20, fontWeight: 700, color: '#8080A8' }}>
                {String(state.lap_count).padStart(2, '0')}
              </div>
            </div>
            {bestLap != null && (
              <div>
                <div style={{ fontSize: 9, fontWeight: 600, letterSpacing: '0.15em', color: '#3A3A5A' }}>BEST</div>
                <div style={{ fontFamily: "'Orbitron', monospace", fontSize: 20, fontWeight: 700, color: '#FFB800' }}>
                  {fmtSec(bestLap)}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Info card */}
        <div style={{
          background: '#0C0C18',
          border: '1px solid #1A1A2E',
          borderRadius: 4,
          padding: '20px 24px',
          position: 'relative',
          overflow: 'hidden',
        }}>
          <div style={{ position: 'absolute', top: 0, left: 0, width: 3, height: '100%', background: statusColor(raceStatus) }} />
          <div style={{ fontSize: 10, fontWeight: 600, letterSpacing: '0.2em', color: '#4A4A6A', marginBottom: 8 }}>
            RACE DATA
          </div>
          <InfoRow
            label="STATUS"
            value={raceStatus}
            valueColor={statusColor(raceStatus)}
          />
          <InfoRow
            label="CONTROLLER"
            value={state.controller_name || '—'}
          />
          <InfoRow
            label="SPEED"
            value={`${state.velocity.toFixed(2)} m/s`}
            live={state.velocity > 0}
          />
          <InfoRow
            label="POSITION"
            value={state.position ? `[${state.position.x.toFixed(2)}, ${state.position.y.toFixed(2)}]` : '—'}
          />
        </div>
      </div>

      {/* ── Lap history ── */}
      <div style={{
        width: '100%', maxWidth: 900,
        background: '#0C0C18',
        border: '1px solid #1A1A2E',
        borderRadius: 4,
        padding: '16px 20px',
        marginBottom: 12,
      }}>
        <div style={{ fontSize: 10, fontWeight: 600, letterSpacing: '0.2em', color: '#4A4A6A', marginBottom: 10 }}>
          LAP HISTORY
        </div>
        {state.lap_times.length === 0 ? (
          <div style={{ fontFamily: "'Barlow Condensed', sans-serif", fontSize: 13, color: '#2A2A45', letterSpacing: '0.1em', padding: '4px 0' }}>
            NO LAPS RECORDED YET
          </div>
        ) : (
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(160px, 1fr))',
            gap: 4,
            maxHeight: 180,
            overflowY: 'auto',
          }}>
            {state.lap_times.map((t, i) => (
              <LapEntry
                key={i}
                lap={i + 1}
                time={t}
                isBest={t === bestLap}
                isCurrent={i === state.lap_times.length - 1 && state.race_running}
              />
            ))}
          </div>
        )}
      </div>

      {/* ── Controls ── */}
      <div style={{
        width: '100%', maxWidth: 900,
        background: '#0C0C18',
        border: '1px solid #1A1A2E',
        borderRadius: 4,
        padding: '16px 20px',
      }}>
        <div style={{ fontSize: 10, fontWeight: 600, letterSpacing: '0.2em', color: '#4A4A6A', marginBottom: 12 }}>
          RACE CONTROL
        </div>

        {/* Row 1: secondary actions */}
        <div style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
          <CtrlBtn label="PAUSE" sub="[P]" path="/api/race/pause"    color="amber" onAction={act} loading={loading} />
          <CtrlBtn label="RESUME" sub="[R]" path="/api/race/resume"  color="green" onAction={act} loading={loading} />
          <CtrlBtn label="RESET LAP" sub="[T]" path="/api/race/reset_lap" color="blue" onAction={act} loading={loading} />
        </div>

        {/* Row 2: high-impact actions */}
        <div style={{ display: 'flex', gap: 8 }}>
          <CtrlBtn label="IGNORE / RESET" sub="[I] · no save" path="/api/race/reset" color="ghost" onAction={act} loading={loading} />
          <CtrlBtn label="END + SAVE" sub="[F] · completes race" path="/api/race/force_complete" color="red" onAction={act} loading={loading} />
        </div>
      </div>

      {/* ── Feedback toast ── */}
      {feedback && (
        <div style={{
          position: 'fixed', bottom: 24, right: 24,
          background: feedback.ok ? '#001A0A' : '#1A0005',
          border: `1px solid ${feedback.ok ? '#00D96A' : '#E8002D'}`,
          borderRadius: 3,
          padding: '10px 16px',
          fontFamily: "'Barlow Condensed', sans-serif",
          fontSize: 13, fontWeight: 600, letterSpacing: '0.08em',
          color: feedback.ok ? '#00D96A' : '#FF3D5A',
          animation: 'slide-in 0.2s ease',
          zIndex: 1000,
          maxWidth: 280,
        }}>
          {feedback.ok ? '✓ ' : '✗ '}{feedback.msg}
        </div>
      )}
    </div>
  );
}
