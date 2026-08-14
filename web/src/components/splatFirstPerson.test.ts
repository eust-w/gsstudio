// @vitest-environment jsdom
import { describe, expect, it } from "vitest";
import * as THREE from "three";

import {
  clampToNavigationBounds,
  initialCameraPose,
  requestPointerLockSafely,
  supportsPointerLock,
  moveOnGround,
  parseViewerMetadata,
} from "./splatFirstPerson";

describe("official GS metadata camera pose", () => {
  it("uses the trainer's center, facing and up direction with the official 70 degree FOV", () => {
    const metadata = parseViewerMetadata({
      center_point: [1, 2, 3],
      facing_direction: [4, 5, 6],
      up_direction: [0, 0, 1],
      human_scale: 1.5,
    });

    expect(metadata).not.toBeNull();
    const pose = initialCameraPose(metadata!);
    expect(pose.position.toArray()).toEqual([1, 2, 3]);
    expect(pose.lookAt.toArray()).toEqual([4, 5, 6]);
    expect(pose.up.toArray()).toEqual([0, 0, 1]);
    expect(pose.fov).toBe(70);
    expect(pose.humanScale).toBe(1.5);
  });

  it("returns no metadata pose for legacy tasks with incomplete files", () => {
    expect(parseViewerMetadata({ center_point: [0, 0, 0] })).toBeNull();
  });
});

describe("first-person movement", () => {
  it("only enables mouse lock when both Pointer Lock APIs are available", () => {
    expect(supportsPointerLock({}, {})).toBe(false);
    expect(supportsPointerLock({ requestPointerLock: () => {} }, {})).toBe(false);
    expect(supportsPointerLock({}, { exitPointerLock: () => {} })).toBe(false);
    expect(
      supportsPointerLock(
        { requestPointerLock: () => {} },
        { exitPointerLock: () => {} },
      ),
    ).toBe(true);

    const standalone: { top?: unknown; parent?: unknown } = {};
    standalone.top = standalone;
    standalone.parent = standalone;
    expect(
      supportsPointerLock(
        { requestPointerLock: () => {} },
        { exitPointerLock: () => {} },
        standalone,
      ),
    ).toBe(true);
    expect(
      supportsPointerLock(
        { requestPointerLock: () => {} },
        { exitPointerLock: () => {} },
        { top: standalone, parent: standalone },
      ),
    ).toBe(false);
  });

  it("handles a rejected Pointer Lock promise without an unhandled browser error", async () => {
    let unavailable = 0;
    requestPointerLockSafely(
      {
        requestPointerLock: () => Promise.reject(new DOMException("invalid document", "WrongDocumentError")),
      },
      () => { unavailable += 1; },
    );

    await Promise.resolve();
    await Promise.resolve();
    expect(unavailable).toBe(1);
  });

  it("moves on an arbitrary ground plane and applies the Shift multiplier", () => {
    const position = new THREE.Vector3(0, 0, 0);
    const direction = new THREE.Vector3(0, -1, 0);
    const up = new THREE.Vector3(0, 0, 1);

    const moved = moveOnGround(position, direction, up, new Set(["KeyW", "ShiftLeft"]), 1, 2);

    expect(moved.distanceTo(position)).toBeCloseTo(6);
    expect(moved.clone().sub(position).dot(up)).toBeCloseTo(0);
    expect(moved.y).toBeCloseTo(-6);
  });

  it("clamps movement to air-wall and camera OBB boundaries", () => {
    const metadata = parseViewerMetadata({
      center_point: [0, 0, 1],
      facing_direction: [0, -1, 1],
      up_direction: [0, 0, 1],
      air_wall: { bbox: [-2, 2, -2, 2, 0, 2] },
      kwargs: { camera_obb: [0, 0, 1, 1, 1, 0] },
    });

    const bounded = clampToNavigationBounds(
      new THREE.Vector3(0, 0, 1),
      new THREE.Vector3(3, 3, 4),
      metadata!,
    );

    expect(bounded.toArray()).toEqual([0, 0, 1]);
  });

  it("does not move or change height when the camera OBB has zero width", () => {
    const metadata = parseViewerMetadata({
      center_point: [2, 3, 4],
      facing_direction: [2, 2, 4],
      up_direction: [0, 0, 1],
      air_wall: { bbox: [-10, 10, -10, 10, -10, 10] },
      kwargs: { camera_obb: [2, 3, 4, 0, 2, 0] },
    });
    const current = new THREE.Vector3(2, 3, 4);

    const bounded = clampToNavigationBounds(current, new THREE.Vector3(3, 3, 9), metadata!);

    expect(bounded.toArray()).toEqual([2, 3, 4]);
  });
});
