import { useEffect, useRef, useState, useCallback } from 'react';

export interface RaceState {
  ros_connected: boolean;
  race_monitor_connected: boolean;
  race_running: boolean;
  race_status: string;
  lap_count: number;
  lap_time: number;
  lap_times: number[];
  controller_name: string;
  position: { x: number; y: number } | null;
  velocity: number;   // m/s
  heading: number;    // radians
  camera_frame: string | null;
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
  controller_name: '',
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

  // Elapsed time tracking — accumulated handles pause/resume correctly
  const raceStartRef    = useRef<number | null>(null); // wall-clock when current segment started
  const lapStartRef     = useRef<number | null>(null);
  const raceAccumRef    = useRef(0);  // ms accumulated across pause segments (race total)
  const lapAccumRef     = useRef(0);  // ms accumulated across pause segments (current lap)
  const prevRunning     = useRef(false);
  const prevLapCount    = useRef(0);
  const isFirstUpdate   = useRef(true);

  const [elapsed, setElapsed]       = useState(0);
  const [lapElapsed, setLapElapsed] = useState(0);

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

  // Track timestamps — skip first message, accumulate across pauses
  useEffect(() => {
    const running = state.race_running;
    const lap     = state.lap_count;
    const now     = Date.now();

    if (isFirstUpdate.current) {
      isFirstUpdate.current = false;
      prevRunning.current   = running;
      prevLapCount.current  = lap;
      return;
    }

    const wasRunning = prevRunning.current;

    if (running && !wasRunning) {
      // Started or resumed — begin a new wall-clock segment
      raceStartRef.current = now;
      lapStartRef.current  = now;
    }

    if (!running && wasRunning) {
      // Paused or stopped — bank elapsed time into accumulators
      if (raceStartRef.current !== null) {
        raceAccumRef.current += now - raceStartRef.current;
        raceStartRef.current  = null;
      }
      if (lapStartRef.current !== null) {
        lapAccumRef.current  += now - lapStartRef.current;
        lapStartRef.current   = null;
      }
    }

    if (lap > prevLapCount.current && running) {
      // New lap started — reset lap timer, keep race total running
      lapStartRef.current  = now;
      lapAccumRef.current  = 0;
    }

    prevRunning.current  = running;
    prevLapCount.current = lap;
  }, [state.race_running, state.lap_count]);

  // Tick at 10 Hz — add live segment to accumulated time
  useEffect(() => {
    const id = setInterval(() => {
      const now = Date.now();
      const liveSeg = raceStartRef.current !== null ? now - raceStartRef.current : 0;
      const liveLap = lapStartRef.current  !== null ? now - lapStartRef.current  : 0;
      setElapsed(raceAccumRef.current + liveSeg);
      setLapElapsed(lapAccumRef.current + liveLap);
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

/** Format seconds → "15.994s" */
export function fmtSec(s: number): string {
  return `${Math.max(0, s).toFixed(3)}s`;
}
