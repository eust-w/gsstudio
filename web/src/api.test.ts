import {describe, expect, it} from "vitest";
import {apiErrorMessage, getArtifactUrl} from "./api";
describe("API boundary", () => {
  it("preserves structured validation messages instead of hiding them behind HTTP status", () => {
    expect(apiErrorMessage({detail:{message:"Runtime is not ready",blockers:[{code:"cuda",message:"CUDA is unavailable"}]}},"Failure")).toBe("Runtime is not ready · CUDA is unavailable");
    expect(apiErrorMessage({detail:[{msg:"At least 12 frames required"}]},"Failure")).toBe("At least 12 frames required");
  });
  it("encodes artifact names without destroying directory separators", () => {
    expect(getArtifactUrl("scene","output/model #1%.spz")).toBe("/api/tasks/scene/artifacts/output/model%20%231%25.spz");
  });
});
