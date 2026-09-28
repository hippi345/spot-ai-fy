import type { KeyboardEvent } from "react";

/** True when Enter should send (not Shift+Enter, not IME composing). */
export function shouldSendChatOnEnter(e: KeyboardEvent): boolean {
  if (e.key !== "Enter") return false;
  if (e.shiftKey) return false;
  const native = e.nativeEvent;
  if (native.isComposing || native.keyCode === 229) return false;
  return true;
}
