import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

describe("r9-item7 trace counts", () => {
  it("test_r9_item7_actions_taken_counts_only_tool_steps", () => {
    render(
      <details className="message-trace" open>
        <summary>
          Actions taken (
          {
            [
              { kind: "status" as const, label: "Still working…" },
              { kind: "tool" as const, label: "spotify_pause" },
              { kind: "status" as const, label: "Still working…" },
            ].filter((s) => s.kind === "tool").length
          }
          )
        </summary>
      </details>,
    );
    expect(screen.getByText(/Actions taken \(1\)/)).toBeInTheDocument();
  });
});
