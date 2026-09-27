import { useEffect, useId, useRef, type ReactNode } from "react";

type Props = {
  open: boolean;
  title: string;
  onClose: () => void;
  children: ReactNode;
};

export function SettingsSheet({ open, title, onClose, children }: Props) {
  const titleId = useId();
  const panelRef = useRef<HTMLDivElement>(null);
  const restoreFocusRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (!open) return undefined;
    restoreFocusRef.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;

    const focusable = () =>
      panelRef.current?.querySelector<HTMLElement>(
        'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
      );

    window.requestAnimationFrame(() => {
      focusable()?.focus();
    });

    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        onClose();
      }
    };
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      restoreFocusRef.current?.focus();
    };
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div className="settings-sheet-root" data-testid="settings-sheet-root">
      <button
        type="button"
        className="settings-sheet-backdrop"
        aria-label="Close settings"
        onClick={onClose}
        data-testid="settings-sheet-backdrop"
      />
      <div
        ref={panelRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        className="settings-sheet glass-panel"
        data-testid="settings-sheet-panel"
      >
        <div className="settings-sheet-head">
          <h2 id={titleId} className="settings-sheet-title">{title}</h2>
          <button
            type="button"
            className="settings-sheet-close secondary"
            aria-label="Close"
            onClick={onClose}
            data-testid="settings-sheet-close"
          >
            ×
          </button>
        </div>
        <div className="settings-sheet-body">{children}</div>
      </div>
    </div>
  );
}
