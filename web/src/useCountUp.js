import { useEffect, useRef, useState } from "react";

// Counts to a value once, on a timer rather than per frame of a transition,
// so the number is readable the whole way up. Honours reduced motion by
// landing on the value immediately.
export default function useCountUp(target, ms = 900) {
  const [value, setValue] = useState(0);
  const raf = useRef(0);

  useEffect(() => {
    if (typeof window === "undefined" ||
        window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      setValue(target);
      return undefined;
    }
    const start = performance.now();
    const tick = (now) => {
      const t = Math.min(1, (now - start) / ms);
      // Ease out, so it decelerates into the final figure instead of stopping.
      setValue(target * (1 - Math.pow(1 - t, 3)));
      if (t < 1) raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf.current);
  }, [target, ms]);

  return value;
}
