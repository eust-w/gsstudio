const WORKER_TERMINATION_MESSAGE = "Worker terminate";

function isExpectedWorkerTermination(reason: unknown) {
  return reason instanceof Error && reason.message === WORKER_TERMINATION_MESSAGE;
}

export function disposeSplatResources(dispose: () => void) {
  const handleUnhandledRejection = (event: PromiseRejectionEvent) => {
    if (isExpectedWorkerTermination(event.reason)) {
      event.preventDefault();
    }
  };

  window.addEventListener("unhandledrejection", handleUnhandledRejection);
  try {
    dispose();
  } finally {
    window.setTimeout(() => {
      window.removeEventListener("unhandledrejection", handleUnhandledRejection);
    }, 0);
  }
}
