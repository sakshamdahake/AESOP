import { Routes, Route } from "react-router-dom";
import MainLayout from "../../layouts/MainLayout";
import Workspace from "./Workspace";

type Props = {
  auth: {
    logout: () => void;
  };
};

function MainPage({ auth }: Props) {
  return (
    <Routes>
      <Route element={<MainLayout onLogout={auth.logout} />}>
        <Route path="/" element={<Workspace />} />
        <Route path=":sessionId" element={<Workspace />} />
      </Route>
    </Routes>
  );
}

export default MainPage;
