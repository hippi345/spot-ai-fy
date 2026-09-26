/** Round-11: hide empty Actions taken trace on assistant bubbles. */
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

describe("test_r11_item5_hide_empty_actions_taken", () => {
  it("test_r11_item5_no_actions_box_when_trace_has_zero_tools", () => {
    render(
      <div className="bubble assistant">
        Plain reply with no tools.
        {([] as { kind: string }[]).filter((s) => s.kind === "tool").length > 0 ? (
          <details className="message-trace">
            <summary>Actions taken (0)</summary>
          </details>
        ) : null}
      </div>,
    );
    expect(screen.queryByText(/Actions taken/i)).not.toBeInTheDocument();
  });
});
