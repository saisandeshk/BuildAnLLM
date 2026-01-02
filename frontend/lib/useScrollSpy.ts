"use client";

import { useEffect, useState } from "react";

export type NavSection = {
  id: string;
  label: string;
};

type ScrollSpyOptions = {
  rootMargin?: string;
};

export function useScrollSpy(sections: NavSection[], options: ScrollSpyOptions = {}) {
  const [activeSection, setActiveSection] = useState(sections[0]?.id ?? "");

  useEffect(() => {
    if (sections.length === 0) {
      return;
    }
    setActiveSection((prev) => prev || sections[0].id);
  }, [sections]);

  useEffect(() => {
    if (sections.length === 0) {
      return;
    }
    const elements = sections
      .map((section) => document.getElementById(section.id))
      .filter((element): element is HTMLElement => Boolean(element));
    if (elements.length === 0) {
      return;
    }
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            setActiveSection(entry.target.id);
          }
        });
      },
      { rootMargin: options.rootMargin ?? "-30% 0px -60% 0px" }
    );
    elements.forEach((element) => observer.observe(element));
    return () => observer.disconnect();
  }, [sections, options.rootMargin]);

  return { activeSection, setActiveSection };
}
