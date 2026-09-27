type IconProps = { className?: string };

export function IconGear({ className }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 24 24" width="20" height="20" aria-hidden="true">
      <path
        fill="currentColor"
        d="M12 8.5a3.5 3.5 0 1 0 0 7 3.5 3.5 0 0 0 0-7zm8.94 4.5 1.33-.77-.3-1.53 1.33-.77-.3-1.53-1.33-.77.3-1.53-1.33-.77.3-1.53-1.33-.77-1.33.77-.3 1.53-1.33.77.3 1.53-1.33.77.3 1.53 1.33.77-.3 1.53 1.33.77-.3 1.53 1.33.77 1.33-.77.3-1.53 1.33-.77-.3-1.53 1.33-.77-.3-1.53-1.33-.77.3-1.53z"
      />
    </svg>
  );
}

export function IconClose({ className }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
      <path
        fill="currentColor"
        d="M6.4 5 5 6.4 10.6 12 5 17.6 6.4 19 12 13.4 17.6 19 19 17.6 13.4 12 19 6.4 17.6 5 12 10.6z"
      />
    </svg>
  );
}

export function IconMinimize({ className }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 12 12" width="12" height="12" aria-hidden="true">
      <rect x="1" y="5.5" width="10" height="1" fill="currentColor" />
    </svg>
  );
}

export function IconMaximize({ className }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 12 12" width="12" height="12" aria-hidden="true">
      <rect x="1.5" y="1.5" width="9" height="9" fill="none" stroke="currentColor" strokeWidth="1" />
    </svg>
  );
}
