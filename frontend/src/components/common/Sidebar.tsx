import { useNavigate } from "react-router-dom";
import type { ChatSession } from "../../api/sessions";

type Props = {
  sessions: ChatSession[];
  loading: boolean;
  onRefresh: () => void;
};

function Sidebar({ sessions, loading }: Props) {
  const navigate = useNavigate();

  return (
    <aside
      style={{
        width: "260px",
        borderRight: "1px solid #ddd",
        padding: "1rem",
      }}
    >
      <button
        style={{ width: "100%", marginBottom: "1rem" }}
        onClick={() => navigate("/chat")}
      >
        New Research
      </button>

      {loading && <p style={{ color: "#666" }}>Loading sessions…</p>}

      {!loading && sessions.length === 0 && (
        <p style={{ fontSize: "0.9rem", color: "#666" }}>
          No research sessions yet
        </p>
      )}

      {sessions.map((s) => (
        <div
          key={s.session_id}
          style={{
            padding: "0.5rem",
            cursor: "pointer",
            borderBottom: "1px solid #eee",
          }}
          onClick={() => navigate(`/chat/${s.session_id}`)}
        >
          {s.title}
        </div>
      ))}
    </aside>
  );
}

export default Sidebar;
