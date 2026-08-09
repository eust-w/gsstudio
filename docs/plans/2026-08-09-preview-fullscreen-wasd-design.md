# Preview Fullscreen and 3D Keyboard Navigation Design

## Goal

Make generated results immediately explorable from the task detail page:

- Prefer the 3D model when a task has one.
- Let Panorama, Video, and 3D Model previews enter browser fullscreen.
- Let users move continuously through a 3D Gaussian Splat scene with WASD.

## Chosen approach

Use the browser Fullscreen API on a shared preview container. If native fullscreen
is unavailable or rejected, fall back to an in-page viewport overlay. Keep the
current tab and viewer instance alive so entering or leaving fullscreen does not
reload the artifact or reset the camera.

## Preview selection

When task data loads, choose the first available preview in this order:

1. 3D Model
2. Video
3. Panorama

After the user changes tabs, refreshes of a pending or running task preserve that
choice while the selected preview remains available.

## Fullscreen interaction

- Every available preview is wrapped in the same fullscreen-capable frame.
- A labelled button in the upper-right corner enters or exits fullscreen.
- Escape exits browser fullscreen through native browser behavior.
- The fallback overlay also exits with Escape.
- The control remains readable over bright and dark preview content.
- Entering 3D fullscreen focuses the viewer so keyboard navigation works
  immediately.

## 3D navigation

The existing OrbitControls mouse interaction remains available. Keyboard movement
is added to the render loop and uses elapsed frame time for stable speed:

- W: move forward along the camera view direction.
- S: move backward.
- A: strafe left.
- D: strafe right.
- Shift: temporarily increase movement speed.

Movement translates both the camera and OrbitControls target, preserving the
current viewing direction. Keyboard input is active only while the viewer is
focused or fullscreen, and never intercepts input from task-name fields or other
form controls. Key state is cleared on blur and component cleanup to prevent
stuck movement.

## User feedback and accessibility

- The 3D viewer is keyboard focusable and has an accessible control label.
- A visible hint documents mouse controls, WASD, and Shift.
- Focused keyboard-control state receives a visible outline.
- Fullscreen buttons expose Enter/Exit labels and pressed state to assistive
  technology.

## Testing

- Component tests cover preview priority and fullscreen controls for all preview
  types.
- Viewer tests cover WASD key registration, Shift acceleration, input-field
  exclusion, and cleanup.
- Production build and existing frontend/backend suites must remain green.
- Public browser acceptance verifies 3D is selected by default, all three preview
  tabs expose fullscreen, native fullscreen toggles, and WASD changes the live
  3D camera without console errors.

