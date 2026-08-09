// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from "vitest";

import { disposeSplatResources, disposeSplatResourcesWhenIdle } from "./splatCleanup";

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

  it("waits for an in-flight Spark sort before terminating its workers", () => {
    vi.useFakeTimers();
    let idle = false;
    const dispose = vi.fn();

    disposeSplatResourcesWhenIdle(() => idle, dispose);
    expect(dispose).not.toHaveBeenCalled();

    vi.advanceTimersByTime(32);
    expect(dispose).not.toHaveBeenCalled();

    idle = true;
    vi.advanceTimersByTime(16);
    expect(dispose).toHaveBeenCalledOnce();
  });
});
