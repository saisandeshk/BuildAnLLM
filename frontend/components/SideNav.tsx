"use client";

import { NavSection } from "../lib/useScrollSpy";

type SideNavProps = {
  sections: NavSection[];
  activeId: string;
  onNavigate?: (id: string) => void;
  ariaLabel?: string;
  title?: string;
};

export default function SideNav({
  sections,
  activeId,
  onNavigate,
  ariaLabel = "Page sections",
  title = "Jump to",
}: SideNavProps) {
  return (
    <nav className="side-nav" aria-label={ariaLabel}>
      <div className="side-nav-title">{title}</div>
      <div className="side-nav-links">
        {sections.map((section) => (
          <a
            key={section.id}
            href={`#${section.id}`}
            className={activeId === section.id ? "active" : ""}
            aria-current={activeId === section.id ? "location" : undefined}
            onClick={() => onNavigate?.(section.id)}
          >
            {section.label}
          </a>
        ))}
      </div>
    </nav>
  );
}
