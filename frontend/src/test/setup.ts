import "@testing-library/jest-dom/vitest";
import "../styles.css";
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

afterEach(() => {
  cleanup();
});
