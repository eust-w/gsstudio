export type IconName = "cube" | "scenes" | "plus" | "system" | "search" | "refresh" | "arrow" | "camera" | "alert" | "check" | "monitor";

const paths: Record<IconName, string> = {
  cube: "m12 2 9 5v10l-9 5-9-5V7l9-5Zm0 10 9-5M12 12 3 7m9 5v10M7.5 4.5l9 5",
  scenes: "M3 7V5a1 1 0 0 1 1-1h5l2 3h9a1 1 0 0 1 1 1v11a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V7Zm0 3h18",
  plus: "M12 5v14M5 12h14",
  system: "M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8ZM9 3l-1 3-3 1-2 3 2 2-1 3 3 2 1 3h4l1-3 3-1 3-3-2-2 1-3-3-2-1-3H9Z",
  search: "M10.5 3a7.5 7.5 0 1 0 0 15 7.5 7.5 0 0 0 0-15ZM16 16l5 5",
  refresh: "M20 7V3m0 4h-4M4 17v4m0-4h4M20 7A9 9 0 0 0 4 8m0 9a9 9 0 0 0 16-1",
  arrow: "M5 12h14m-6-6 6 6-6 6",
  camera: "M3 7h4l2-3h6l2 3h4v13H3V7Zm9 2a4 4 0 1 0 0 8 4 4 0 0 0 0-8Z",
  alert: "M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18Zm0 4v6m0 3v.1",
  check: "m5 12 4 4L19 6",
  monitor: "M3 3h18v14H3V3Zm9 14v4m-5 0h10",
};

export default function Icon({name, size = 20}: {name: IconName; size?: number}) {
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={paths[name]} /></svg>;
}
