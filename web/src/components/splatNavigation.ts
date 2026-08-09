import * as THREE from "three";

const BOOST_MULTIPLIER = 3;

export function isEditableTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  return (
    target.matches("input, textarea, select, button") ||
    target.isContentEditable ||
    target.contentEditable === "true" ||
    target.getAttribute("contenteditable") === "true"
  );
}

export function moveCamera(
  camera: THREE.Camera,
  target: THREE.Vector3,
  keys: ReadonlySet<string>,
  deltaSeconds: number,
  speed: number,
): boolean {
  const movement = new THREE.Vector3();
  const forward = camera.getWorldDirection(new THREE.Vector3());
  const right = new THREE.Vector3(1, 0, 0).applyQuaternion(camera.quaternion).normalize();

  if (keys.has("KeyW")) movement.add(forward);
  if (keys.has("KeyS")) movement.sub(forward);
  if (keys.has("KeyD")) movement.add(right);
  if (keys.has("KeyA")) movement.sub(right);
  if (movement.lengthSq() === 0) return false;

  const boosted = keys.has("ShiftLeft") || keys.has("ShiftRight");
  const distance = speed * deltaSeconds * (boosted ? BOOST_MULTIPLIER : 1);
  movement.normalize().multiplyScalar(distance);
  camera.position.add(movement);
  target.add(movement);
  return true;
}
