import { useEffect, useRef, useState, useCallback } from 'react';

export interface RaceState {
  ros_connected: boolean;
  race_monitor_connected: boolean;
  race_running: boolean;
  race_status: string;
  lap_count: number;
  lap_time: number;
  lap_times: number[];
  position: { x: number; y: number } | null;
  velocity: number;   // m/s
  heading: number;    // radians
  camera_frame: string | null;  // base64 JPEG or null
  ts: number;
}

const DEFAULT_STATE: RaceState = {
  ros_connected: false,
  race_monitor_connected: false,
  race_running: false,
  race_status: 'Waiting...',
  lap_count: 0,
  lap_time: 0,
  lap_times: [],
  position: null,
  velocity: 0,
  heading: 0,
  camera_frame: null,
  ts: 0,
};

const WS_URL = `ws://${window.location.hostname}:8082/ws`;

export function useRaceData() {
  const [state, setState] = useState<RaceState>(DEFAULT_STATE);
  const [wsConnected, setWsConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Elapsed time tracking
  const raceStartRef = useRef<number | null>(null);
  const lapStartRef  = useRef<number | null>(null);
  const prevRunning  = useRef(false);
  const prevLapCount = useRef(0);

  const [elapsed, setElapsed]    = useState(0); // ms since race start
  const [lapElapsed, setLapElapsed] = useState(0); // ms since current lap start

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      setWsConnected(true);
      if (reconnectRef.current) clearTimeout(reconnectRef.current);
    };

    ws.onmessage = (ev) => {
      try {
        const data: RaceState = JSON.parse(ev.data);
        setState(data);
      } catch {
        // ignore parse errors
      }
    };

    ws.onclose = () => {
      setWsConnected(false);
      reconnectRef.current = setTimeout(connect, 2000);
    };

    ws.onerror = () => {
      ws.close();
    };
  }, []);

  useEffect(() => {
    connect();
    return () => {
      if (reconnectRef.current) clearTimeout(reconnectRef.current);
      wsRef.current?.close();
    };
  }, [connect]);

  // Track start timestamps
  useEffect(() => {
    const running = state.race_running;
    const lap = state.lap_count;

    if (running && !prevRunning.current) {
      raceStartRef.current = Date.now();
      lapStartRef.current  = Date.now();
    }
    if (!running && prevRunning.current) {
      raceStartRef.current = null;
      lapStartRef.current  = null;
    }
    if (lap > prevLapCount.current && running) {
      lapStartRef.current = Date.now();
    }

    prevRunning.current  = running;
    prevLapCount.current = lap;
  }, [state.race_running, state.lap_count]);

  // Tick local timers at 10 Hz
  useEffect(() => {
    const id = setInterval(() => {
      setElapsed(raceStartRef.current ? Date.now() - raceStartRef.current : 0);
      setLapElapsed(lapStartRef.current ? Date.now() - lapStartRef.current : 0);
    }, 100);
    return () => clearInterval(id);
  }, []);

  return { state, wsConnected, elapsed, lapElapsed };
}

/** Format milliseconds → "MM:SS.mmm" */
export function fmtMs(ms: number): string {
  const total = Math.max(0, ms);
  const mins  = Math.floor(total / 60000);
  const secs  = Math.floor((total % 60000) / 1000);
  const millis = Math.floor((total % 1000) / 10); // centiseconds
  return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}.${String(millis).padStart(2, '0')}`;
}

/** Format seconds → "MM:SS.mmm" */
export function fmtSec(s: number): string {
  return fmtMs(s * 1000);
}
