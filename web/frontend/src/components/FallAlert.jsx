import { useEffect, useRef } from "react";

/**
 * Red flashing banner + beep when fall_alert is true.
 * Props: active (boolean)
 */
export default function FallAlert({ active }) {
  const beepRef = useRef(null);

  // One-shot beep via Web Audio API on rising edge
  useEffect(() => {
    if (!active) return;
    try {
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.type = "square";
      osc.frequency.setValueAtTime(880, ctx.currentTime);
      gain.gain.setValueAtTime(0.3, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.4);
      osc.start(ctx.currentTime);
      osc.stop(ctx.currentTime + 0.4);
    } catch {
      // AudioContext not available — silent
    }
  }, [active]);

  if (!active) return null;

  return (
    <div className="animate-flash flex items-center gap-3 bg-red-700 border border-red-500 text-white font-bold rounded-lg px-4 py-3 shadow-lg shadow-red-900/50">
      <svg
        className="w-5 h-5 flex-shrink-0"
        fill="currentColor"
        viewBox="0 0 20 20"
      >
        <path
          fillRule="evenodd"
          d="M8.485 2.495c.673-1.167 2.357-1.167 3.03 0l6.28 10.875c.673 1.167-.17 2.625-1.516 2.625H3.72c-1.347 0-2.189-1.458-1.515-2.625L8.485 2.495zM10 5a.75.75 0 01.75.75v3.5a.75.75 0 01-1.5 0v-3.5A.75.75 0 0110 5zm0 9a1 1 0 100-2 1 1 0 000 2z"
          clipRule="evenodd"
        />
      </svg>
      FALL DETECTED
    </div>
  );
}
