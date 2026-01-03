"use client";

import { useEffect } from "react";
import { init } from "@plausible-analytics/tracker";
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
    init({ domain: "buildanllm.com" });
    initialized = true;
  }, [isDemo]);

  return null;
}
