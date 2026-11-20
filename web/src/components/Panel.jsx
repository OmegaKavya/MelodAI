import React from "react";
import useReveal from "../useReveal";

// Every card on the page is one of these, so the glass treatment and the
// reveal are defined once rather than remembered eight times.
export default function Panel({ className = "", children, ...rest }) {
  const [ref, reveal] = useReveal();
  return (
    <section ref={ref} className={`panel ${reveal} ${className}`.trim()} {...rest}>
      {children}
    </section>
  );
}
