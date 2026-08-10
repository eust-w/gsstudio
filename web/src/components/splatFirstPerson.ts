import * as THREE from "three";

const BOOST_MULTIPLIER = 3;

/**
 * Some embedded and automated Chromium contexts expose keyboard focus but not
 * the Pointer Lock API.  Keep WASD available there without invoking Three's
 * error-reporting pointer-lock path.
 */
export function supportsPointerLock(
  target: { requestPointerLock?: unknown },
  ownerDocument: { exitPointerLock?: unknown },
  browsingContext?: { top?: unknown; parent?: unknown },
): boolean {
  // Pointer Lock is not reliably permitted for a nested/hosted document.
  // In particular, the browser automation host exposes the API surface but
  // logs a WrongDocumentError before a catch handler can suppress it.
  if (
    browsingContext
    && (browsingContext.top !== browsingContext || browsingContext.parent !== browsingContext)
  ) {
    return false;
  }
  return typeof target.requestPointerLock === "function"
    && typeof ownerDocument.exitPointerLock === "function";
}

/**
 * Modern Chromium can reject Pointer Lock as a Promise (rather than throw
 * synchronously) when a page lives in an embedded document. Three's
 * PointerLockControls ignores that return value, leaving an unhandled
 * WrongDocumentError in the console. Request it directly and consume both
 * failure forms so focused WASD remains a quiet fallback.
 */
export function requestPointerLockSafely(
  target: { requestPointerLock?: () => unknown },
  onUnavailable: () => void,
): void {
  if (typeof target.requestPointerLock !== "function") {
    onUnavailable();
    return;
  }
  try {
    Promise.resolve(target.requestPointerLock()).catch(onUnavailable);
  } catch {
    onUnavailable();
  }
}

export interface ViewerMetadata {
  centerPoint: THREE.Vector3;
  facingDirection: THREE.Vector3;
  upDirection: THREE.Vector3;
  humanScale: number;
  cameraObb?: [number, number, number, number, number, number];
  airWall?: [number, number, number, number, number, number];
}

export interface InitialCameraPose {
  position: THREE.Vector3;
  lookAt: THREE.Vector3;
  up: THREE.Vector3;
  fov: number;
  humanScale: number;
}

function toVector(value: unknown): THREE.Vector3 | null {
  if (!Array.isArray(value) || value.length !== 3 || !value.every((entry) => typeof entry === "number" && Number.isFinite(entry))) {
    return null;
  }
  return new THREE.Vector3(value[0], value[1], value[2]);
}

function toBounds(value: unknown): [number, number, number, number, number, number] | undefined {
  if (!Array.isArray(value) || value.length < 6) return undefined;
  const values = value.slice(0, 6);
  if (!values.every((entry) => typeof entry === "number" && Number.isFinite(entry))) return undefined;
  return values as [number, number, number, number, number, number];
}

export function parseViewerMetadata(value: unknown): ViewerMetadata | null {
  if (!value || typeof value !== "object") return null;
  const raw = value as Record<string, unknown>;
  const centerPoint = toVector(raw.center_point);
  const facingDirection = toVector(raw.facing_direction);
  const upDirection = toVector(raw.up_direction);
  if (!centerPoint || !facingDirection || !upDirection || upDirection.lengthSq() === 0) return null;

  const kwargs = raw.kwargs && typeof raw.kwargs === "object" ? raw.kwargs as Record<string, unknown> : {};
  const airWall = raw.air_wall && typeof raw.air_wall === "object" ? raw.air_wall as Record<string, unknown> : {};
  return {
    centerPoint,
    facingDirection,
    upDirection: upDirection.normalize(),
    humanScale: typeof raw.human_scale === "number" && raw.human_scale > 0 ? raw.human_scale : 1,
    cameraObb: toBounds(kwargs.camera_obb),
    airWall: toBounds(airWall.bbox),
  };
}

export function initialCameraPose(metadata: ViewerMetadata): InitialCameraPose {
  return {
    position: metadata.centerPoint.clone(),
    // This mirrors HY-World's `show_gs.py`: center is the camera position and
    // facing_direction is the camera look-at target.
    lookAt: metadata.facingDirection.clone(),
    up: metadata.upDirection.clone(),
    fov: 70,
    humanScale: metadata.humanScale,
  };
}

export function moveOnGround(
  position: THREE.Vector3,
  worldDirection: THREE.Vector3,
  up: THREE.Vector3,
  keys: ReadonlySet<string>,
  deltaSeconds: number,
  humanScale: number,
): THREE.Vector3 {
  const normalizedUp = up.clone().normalize();
  const forward = worldDirection.clone().projectOnPlane(normalizedUp);
  if (forward.lengthSq() === 0) return position.clone();
  forward.normalize();
  const right = new THREE.Vector3().crossVectors(forward, normalizedUp).normalize();
  const movement = new THREE.Vector3();
  if (keys.has("KeyW")) movement.add(forward);
  if (keys.has("KeyS")) movement.sub(forward);
  if (keys.has("KeyD")) movement.add(right);
  if (keys.has("KeyA")) movement.sub(right);
  if (movement.lengthSq() === 0) return position.clone();

  const boosted = keys.has("ShiftLeft") || keys.has("ShiftRight");
  const distance = Math.max(humanScale, 0.05) * deltaSeconds * (boosted ? BOOST_MULTIPLIER : 1);
  return position.clone().add(movement.normalize().multiplyScalar(distance));
}

function clampAirWall(position: THREE.Vector3, airWall?: ViewerMetadata["airWall"]): THREE.Vector3 {
  if (!airWall) return position.clone();
  const [xMin, xMax, yMin, yMax, zMin, zMax] = airWall;
  return new THREE.Vector3(
    THREE.MathUtils.clamp(position.x, Math.min(xMin, xMax), Math.max(xMin, xMax)),
    THREE.MathUtils.clamp(position.y, Math.min(yMin, yMax), Math.max(yMin, yMax)),
    THREE.MathUtils.clamp(position.z, Math.min(zMin, zMax), Math.max(zMin, zMax)),
  );
}

function isInsideCameraObb(position: THREE.Vector3, metadata: ViewerMetadata): boolean {
  const obb = metadata.cameraObb;
  if (!obb) return true;
  const [cx, cy, cz, halfWidth, halfHeight, angle] = obb;
  const up = metadata.upDirection;
  const fallback = Math.abs(up.x) < 0.9
    ? new THREE.Vector3(1, 0, 0)
    : new THREE.Vector3(0, 1, 0);
  const referenceRight = new THREE.Vector3().crossVectors(fallback, up).normalize();
  const referenceForward = new THREE.Vector3().crossVectors(up, referenceRight).normalize();
  const delta = position.clone().sub(new THREE.Vector3(cx, cy, cz)).projectOnPlane(up);
  const u = delta.dot(referenceRight);
  const v = delta.dot(referenceForward);
  const localWidth = u * Math.cos(angle) + v * Math.sin(angle);
  const localHeight = -u * Math.sin(angle) + v * Math.cos(angle);
  return Math.abs(localWidth) <= halfWidth && Math.abs(localHeight) <= halfHeight;
}

export function clampToNavigationBounds(
  currentPosition: THREE.Vector3,
  candidatePosition: THREE.Vector3,
  metadata: ViewerMetadata,
): THREE.Vector3 {
  const bounded = clampAirWall(candidatePosition, metadata.airWall);
  if (isInsideCameraObb(bounded, metadata)) return bounded;

  // The trainer OBB bounds only the ground plane.  Preserve a vertically
  // clamped move while rejecting the unsafe horizontal component.
  const up = metadata.upDirection;
  const verticalDistance = bounded.clone().sub(currentPosition).dot(up);
  return currentPosition.clone().addScaledVector(up, verticalDistance);
}
