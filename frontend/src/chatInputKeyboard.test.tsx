import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { useRef, useState } from "react";

import { shouldSendChatOnEnter } from "./lib/chatInputKeyboard";

function ChatInputProbe({ onSend }: { onSend: () => void }) {
  const [value, setValue] = useState("");
  const composingRef = useRef(false);
  return (
    <textarea
      data-testid="chat"
      value={value}
      onChange={(e) => setValue(e.target.value)}
      onCompositionStart={() => {
        composingRef.current = true;
      }}
      onCompositionEnd={() => {
        composingRef.current = false;
      }}
      onKeyDown={(e) => {
        if (shouldSendChatOnEnter(e) && !composingRef.current) {
          e.preventDefault();
          onSend();
        }
      }}
    />
  );
}

describe("chat input keyboard", () => {
  it("Enter sends when not composing", () => {
    const send = vi.fn();
    render(<ChatInputProbe onSend={send} />);
    const el = screen.getByTestId("chat");
    fireEvent.keyDown(el, { key: "Enter", shiftKey: false });
    expect(send).toHaveBeenCalledTimes(1);
  });

  it("Shift+Enter does not send", () => {
    const send = vi.fn();
    render(<ChatInputProbe onSend={send} />);
    const el = screen.getByTestId("chat");
    fireEvent.keyDown(el, { key: "Enter", shiftKey: true });
    expect(send).not.toHaveBeenCalled();
  });

  it("does not send while IME composing", () => {
    const send = vi.fn();
    render(<ChatInputProbe onSend={send} />);
    const el = screen.getByTestId("chat");
    fireEvent.compositionStart(el);
    fireEvent.keyDown(el, { key: "Enter", isComposing: true, keyCode: 229 });
    expect(send).not.toHaveBeenCalled();
    fireEvent.compositionEnd(el);
    fireEvent.keyDown(el, { key: "Enter" });
    expect(send).toHaveBeenCalledTimes(1);
  });
});
