import { render, screen, fireEvent } from "@testing-library/react";
import { vi } from "vitest";

import Modal from "../../components/Modal";

describe("Modal", () => {
  it("renders nothing when closed", () => {
    const { container } = render(
      <Modal isOpen={false} onClose={() => {}} title="Test Modal">
        <p>Content</p>
      </Modal>
    );
    expect(container.firstChild).toBeNull();
  });

  it("renders title and content when open", () => {
    render(
      <Modal isOpen={true} onClose={() => {}} title="Test Modal">
        <p>Modal content here</p>
      </Modal>
    );

    expect(screen.getByRole("dialog")).toBeInTheDocument();
    expect(screen.getByText("Test Modal")).toBeInTheDocument();
    expect(screen.getByText("Modal content here")).toBeInTheDocument();
  });

  it("calls onClose when close button is clicked", () => {
    const onCloseMock = vi.fn();
    render(
      <Modal isOpen={true} onClose={onCloseMock} title="Test Modal">
        <p>Content</p>
      </Modal>
    );

    const closeButton = screen.getByRole("button", { name: "Close modal" });
    fireEvent.click(closeButton);
    expect(onCloseMock).toHaveBeenCalledTimes(1);
  });

  it("calls onClose when backdrop is clicked", () => {
    const onCloseMock = vi.fn();
    render(
      <Modal isOpen={true} onClose={onCloseMock} title="Test Modal">
        <p>Content</p>
      </Modal>
    );

    // Click on the backdrop (the outermost element)
    const backdrop = screen.getByRole("dialog").parentElement;
    if (backdrop) {
      fireEvent.click(backdrop);
      expect(onCloseMock).toHaveBeenCalledTimes(1);
    }
  });

  it("calls onClose when Escape key is pressed", () => {
    const onCloseMock = vi.fn();
    render(
      <Modal isOpen={true} onClose={onCloseMock} title="Test Modal">
        <p>Content</p>
      </Modal>
    );

    fireEvent.keyDown(document, { key: "Escape" });
    expect(onCloseMock).toHaveBeenCalledTimes(1);
  });

  it("does not call onClose for non-Escape keys", () => {
    const onCloseMock = vi.fn();
    render(
      <Modal isOpen={true} onClose={onCloseMock} title="Test Modal">
        <p>Content</p>
      </Modal>
    );

    fireEvent.keyDown(document, { key: "Enter" });
    fireEvent.keyDown(document, { key: "a" });
    expect(onCloseMock).not.toHaveBeenCalled();
  });

  it("has proper accessibility attributes", () => {
    render(
      <Modal isOpen={true} onClose={() => {}} title="Accessible Modal">
        <p>Content</p>
      </Modal>
    );

    const dialog = screen.getByRole("dialog");
    expect(dialog).toHaveAttribute("aria-modal", "true");
    expect(dialog).toHaveAttribute("aria-labelledby", "modal-title");
    expect(screen.getByText("Accessible Modal")).toHaveAttribute("id", "modal-title");
  });
});
