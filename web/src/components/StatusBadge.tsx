interface StatusBadgeProps {
  status: string;
}

const STATUS_COLORS: Record<string, string> = {
  pending: "#6b7280",
  running: "#3b82f6",
  completed: "#22c55e",
  failed: "#ef4444",
  cancelled: "#f97316",
  skipped: "#9ca3af",
};

export default function StatusBadge({ status }: StatusBadgeProps) {
  const color = STATUS_COLORS[status] || "#6b7280";
  return (
    <span
      className={`status-badge status-${status}`}
      style={{ backgroundColor: color + "22", color, borderColor: color }}
    >
      {status}
    </span>
  );
}
