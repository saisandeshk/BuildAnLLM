"use client";

import { useEffect } from "react";
import { useDemoMode } from "../lib/demo";

let initialized = false;

export default function PlausibleAnalytics() {
  const isDemo = useDemoMode();

  useEffect(() => {
    if (!isDemo) {
      return;
    }
    if (initialized) {
      return;
    }
    let active = true;
    import("@plausible-analytics/tracker")
      .then(({ init }) => {
        if (!active || initialized) {
          return;
        }
        init({ domain: "buildanllm.com" });
        initialized = true;
      })
      .catch(() => {
        // Swallow analytics load failures to keep render clean.
      });
    return () => {
      active = false;
    };
  }, [isDemo]);

  return null;
}
