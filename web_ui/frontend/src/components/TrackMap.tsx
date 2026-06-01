import { useEffect, useRef } from 'react';

interface Point { x: number; y: number; }
interface Props {
  position: Point | null;
  heading: number;
  trail: Point[];
}

const TRAIL_MAX = 500;

export default function TrackMap({ position, heading, trail }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    if (trail.length < 2) {
      // No data yet — draw placeholder
      ctx.strokeStyle = '#222';
      ctx.lineWidth = 1;
      ctx.strokeRect(1, 1, W - 2, H - 2);
      ctx.fillStyle = '#444';
      ctx.font = '12px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Waiting for position data…', W / 2, H / 2);
      return;
    }

    // Compute bounding box with padding
    const xs = trail.map(p => p.x);
    const ys = trail.map(p => p.y);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const pad = 20;
    const rangeX = (maxX - minX) || 10;
    const rangeY = (maxY - minY) || 10;
    const scaleX = (W - pad * 2) / rangeX;
    const scaleY = (H - pad * 2) / rangeY;
    const scale  = Math.min(scaleX, scaleY);
    const offX   = (W - rangeX * scale) / 2 - minX * scale;
    const offY   = (H - rangeY * scale) / 2 - minY * scale;

    const toCanvas = (p: Point) => ({
      x: p.x * scale + offX,
      y: H - (p.y * scale + offY),   // flip Y axis
    });

    // Draw trail
    ctx.beginPath();
    const p0 = toCanvas(trail[0]);
    ctx.moveTo(p0.x, p0.y);
    for (let i = 1; i < trail.length; i++) {
      // Colour shifts from dim → bright as trail ages
      const t = i / trail.length;
      ctx.strokeStyle = `rgba(0, 212, 255, ${0.15 + t * 0.85})`;
      ctx.lineWidth = 1.5 + t;
      const pt = toCanvas(trail[i]);
      ctx.lineTo(pt.x, pt.y);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(pt.x, pt.y);
    }

    // Draw car marker
    if (position) {
      const cp = toCanvas(position);
      const arrow = 10;
      ctx.save();
      ctx.translate(cp.x, cp.y);
      ctx.rotate(-heading);
      // Body
      ctx.fillStyle = '#e8000d';
      ctx.shadowColor = '#e8000d';
      ctx.shadowBlur = 14;
      ctx.beginPath();
      ctx.moveTo(0, -arrow);
      ctx.lineTo(arrow * 0.5, arrow * 0.5);
      ctx.lineTo(0, 0);
      ctx.lineTo(-arrow * 0.5, arrow * 0.5);
      ctx.closePath();
      ctx.fill();
      ctx.shadowBlur = 0;
      ctx.restore();
    }
  }, [trail, position, heading]);

  return (
    <canvas
      ref={canvasRef}
      width={320}
      height={220}
      className="w-full h-full"
      style={{ display: 'block' }}
    />
  );
}

/** Append a position to the trail, capped at TRAIL_MAX. */
export function appendTrail(trail: Point[], pos: Point): Point[] {
  const next = [...trail, pos];
  return next.length > TRAIL_MAX ? next.slice(next.length - TRAIL_MAX) : next;
}
