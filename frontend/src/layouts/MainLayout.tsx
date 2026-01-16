import { useEffect, useState } from "react";
import { Outlet } from "react-router-dom";
import TopBar from "../components/common/TopBar";
import Sidebar from "../components/common/Sidebar";
import { listSessions } from "../api/sessions";
import type { SessionSummary } from "../api/types";

type OutletContextType = {
  refreshSessions: () => void;
};

type Props = {
  onLogout: () => void;
};

function MainLayout({ onLogout }: Props) {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [loading, setLoading] = useState(false);

  const loadSessions = async () => {
    setLoading(true);
    const res = await listSessions();
    setSessions(res.sessions);
    setLoading(false);
  };

  useEffect(() => {
    loadSessions();
  }, []);

  return (
    <div style={{ height: "100vh", display: "flex", flexDirection: "column" }}>
      <TopBar onLogout={onLogout} />

      <div style={{ display: "flex", flex: 1 }}>
        <Sidebar
          sessions={sessions}
          loading={loading}
          onRefresh={loadSessions}
        />

        <main style={{ flex: 1, padding: "2rem", overflowY: "auto" }}>
          {/* 👇 THIS is the key */}
          <Outlet context={{ refreshSessions: loadSessions }} />
        </main>
      </div>
    </div>
  );
}

export type { OutletContextType };
export default MainLayout;
