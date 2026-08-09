// @vitest-environment jsdom
import { describe, expect, it } from "vitest";
import * as THREE from "three";

import { isEditableTarget, moveCamera } from "./splatNavigation";

function createRig() {
  const camera = new THREE.PerspectiveCamera();
  camera.position.set(0, 0, 5);
  camera.lookAt(0, 0, 0);
  camera.updateMatrixWorld();
  return { camera, target: new THREE.Vector3(0, 0, 0) };
}

describe("moveCamera", () => {
  it("moves forward and translates the orbit target by the same amount", () => {
    const { camera, target } = createRig();
    const startCamera = camera.position.clone();
    const startTarget = target.clone();

    moveCamera(camera, target, new Set(["KeyW"]), 1, 2);

    const cameraMovement = camera.position.clone().sub(startCamera);
    const targetMovement = target.clone().sub(startTarget);
    expect(cameraMovement.length()).toBeCloseTo(2);
    expect(targetMovement.distanceTo(cameraMovement)).toBeCloseTo(0);
    expect(camera.position.z).toBeCloseTo(3);
  });

  it("supports reverse and camera-relative strafing", () => {
    const backward = createRig();
    moveCamera(backward.camera, backward.target, new Set(["KeyS"]), 1, 2);
    expect(backward.camera.position.z).toBeCloseTo(7);

    const left = createRig();
    moveCamera(left.camera, left.target, new Set(["KeyA"]), 1, 2);
    expect(left.camera.position.x).toBeCloseTo(-2);

    const right = createRig();
    moveCamera(right.camera, right.target, new Set(["KeyD"]), 1, 2);
    expect(right.camera.position.x).toBeCloseTo(2);
  });

  it("normalizes diagonal movement and accelerates while Shift is held", () => {
    const diagonal = createRig();
    const diagonalStart = diagonal.camera.position.clone();
    moveCamera(diagonal.camera, diagonal.target, new Set(["KeyW", "KeyD"]), 1, 2);
    expect(diagonal.camera.position.distanceTo(diagonalStart)).toBeCloseTo(2);

    const accelerated = createRig();
    const acceleratedStart = accelerated.camera.position.clone();
    moveCamera(accelerated.camera, accelerated.target, new Set(["KeyW", "ShiftLeft"]), 1, 2);
    expect(accelerated.camera.position.distanceTo(acceleratedStart)).toBeCloseTo(6);
  });
});

describe("isEditableTarget", () => {
  it("ignores keyboard movement from form fields and editable elements", () => {
    expect(isEditableTarget(document.createElement("input"))).toBe(true);
    expect(isEditableTarget(document.createElement("textarea"))).toBe(true);
    const editable = document.createElement("div");
    editable.contentEditable = "true";
    expect(isEditableTarget(editable)).toBe(true);
    expect(isEditableTarget(document.createElement("canvas"))).toBe(false);
  });
});
