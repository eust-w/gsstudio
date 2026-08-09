const WORKER_TERMINATION_MESSAGE = "Worker terminate";

function isExpectedWorkerTermination(reason: unknown) {
  return reason instanceof Error && reason.message === WORKER_TERMINATION_MESSAGE;
}

type WorkerWithMessages = { messages: Record<number, unknown> } | null;

export function areSplatWorkersIdle(...workers: WorkerWithMessages[]) {
  return workers.every((worker) => worker === null || Object.keys(worker.messages).length === 0);
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

export function disposeSplatResourcesWhenIdle(isIdle: () => boolean, dispose: () => void) {
  const disposeOrWait = () => {
    if (isIdle()) {
      disposeSplatResources(dispose);
      return;
    }
    window.setTimeout(disposeOrWait, 16);
  };

  disposeOrWait();
}
