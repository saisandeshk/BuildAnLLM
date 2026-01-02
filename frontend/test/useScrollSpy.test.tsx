import { act, render, screen } from "@testing-library/react";

import { useScrollSpy } from "../lib/useScrollSpy";

class MockIntersectionObserver {
  static instances: MockIntersectionObserver[] = [];
  callback: IntersectionObserverCallback;
  elements: Element[] = [];

  constructor(callback: IntersectionObserverCallback) {
    this.callback = callback;
    MockIntersectionObserver.instances.push(this);
  }

  observe(element: Element) {
    this.elements.push(element);
  }

  disconnect() {}

  trigger(target: Element) {
    const entry = {
      isIntersecting: true,
      target,
    } as IntersectionObserverEntry;
    this.callback([entry], this as unknown as IntersectionObserver);
  }
}

function SpyTest({ ids }: { ids: string[] }) {
  const sections = ids.map((id) => ({ id, label: id.toUpperCase() }));
  const { activeSection } = useScrollSpy(sections);
  return <div data-testid="active">{activeSection}</div>;
}

describe("useScrollSpy", () => {
  beforeEach(() => {
    MockIntersectionObserver.instances = [];
    vi.stubGlobal("IntersectionObserver", MockIntersectionObserver);
  });

  it("defaults to the first section", () => {
    render(<SpyTest ids={["alpha", "beta"]} />);
    expect(screen.getByTestId("active")).toHaveTextContent("alpha");
  });

  it("updates when an observed section intersects", () => {
    document.body.innerHTML = `
      <section id="alpha"></section>
      <section id="beta"></section>
    `;

    render(<SpyTest ids={["alpha", "beta"]} />);
    const observer = MockIntersectionObserver.instances[0];
    const beta = document.getElementById("beta") as Element;

    act(() => {
      observer.trigger(beta);
    });

    expect(screen.getByTestId("active")).toHaveTextContent("beta");
  });
});
