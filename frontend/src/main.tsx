import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { isDesktopShell } from "./lib/api";
import "./styles.css";

if (isDesktopShell()) {
  document.body.classList.add("desktop-shell");
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>
);
