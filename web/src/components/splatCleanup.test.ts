// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from "vitest";

import { disposeSplatResources } from "./splatCleanup";

afterEach(() => {
  vi.useRealTimers();
});

describe("disposeSplatResources", () => {
  it("suppresses Spark worker termination rejections during cleanup", () => {
    vi.useFakeTimers();
    const dispose = vi.fn(() => {
      const event = new Event("unhandledrejection", { cancelable: true }) as PromiseRejectionEvent;
      Object.defineProperty(event, "reason", { value: new Error("Worker terminate") });
      window.dispatchEvent(event);
      expect(event.defaultPrevented).toBe(true);
    });

    disposeSplatResources(dispose);
    expect(dispose).toHaveBeenCalledOnce();

    vi.runAllTimers();
    const later = new Event("unhandledrejection", { cancelable: true }) as PromiseRejectionEvent;
    Object.defineProperty(later, "reason", { value: new Error("Worker terminate") });
    window.dispatchEvent(later);
    expect(later.defaultPrevented).toBe(false);
  });

  it("does not hide unrelated cleanup failures", () => {
    const event = new Event("unhandledrejection", { cancelable: true }) as PromiseRejectionEvent;
    Object.defineProperty(event, "reason", { value: new Error("Unexpected render failure") });

    disposeSplatResources(() => window.dispatchEvent(event));

    expect(event.defaultPrevented).toBe(false);
  });
});
