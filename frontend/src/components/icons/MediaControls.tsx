type IconProps = {
  className?: string;
};

export function IconPrevious({ className = "np-icon" }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 24 24" width="20" height="20" aria-hidden="true">
      <path
        fill="currentColor"
        d="M6 6h2v12H6V6zm3.5 6 8.5 6V6l-8.5 6z"
      />
    </svg>
  );
}

export function IconNext({ className = "np-icon" }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 24 24" width="20" height="20" aria-hidden="true">
      <path
        fill="currentColor"
        d="M16 6h2v12h-2V6zM6 6l8.5 6L6 18V6z"
      />
    </svg>
  );
}

export function IconPlay({ className = "np-icon" }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 24 24" width="22" height="22" aria-hidden="true">
      <path fill="currentColor" d="M8 5v14l11-7L8 5z" />
    </svg>
  );
}

export function IconPause({ className = "np-icon" }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 24 24" width="22" height="22" aria-hidden="true">
      <path fill="currentColor" d="M6 5h4v14H6V5zm8 0h4v14h-4V5z" />
    </svg>
  );
}
