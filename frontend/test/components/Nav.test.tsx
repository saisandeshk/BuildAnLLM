import { render, screen } from "@testing-library/react";
import type { ReactNode } from "react";

import Nav from "../../components/Nav";
import { usePathname } from "next/navigation";

vi.mock("next/navigation", () => ({
  usePathname: vi.fn(),
}));

vi.mock("next/link", () => ({
  default: ({ href, className, children }: { href: string; className?: string; children: ReactNode }) => (
    <a href={href} className={className}>
      {children}
    </a>
  ),
}));

describe("Nav", () => {
  const mockedUsePathname = usePathname as unknown as ReturnType<typeof vi.fn>;

  it("highlights the active link", () => {
    mockedUsePathname.mockReturnValue("/pretrain");
    render(<Nav />);

    const activeLink = screen.getByText("Pre-Training");
    expect(activeLink.className).toContain("active");
  });

  it("marks playground as active when on playground routes", () => {
    mockedUsePathname.mockReturnValue("/playground");
    const { container } = render(<Nav />);

    const dropdown = container.querySelector(".nav-dropdown");
    const trigger = container.querySelector(".nav-dropdown-trigger");
    expect(dropdown?.className).toContain("active");
    expect(trigger?.className).toContain("active");
  });
});
