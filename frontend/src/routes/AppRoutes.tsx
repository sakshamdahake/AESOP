import { Routes, Route, Navigate } from "react-router-dom";
import LoginPage from "../pages/Login/LoginPage";
import MainPage from "../pages/Main/MainPage";

type Auth = {
  isAuthed: boolean;
  login: () => void;
  logout: () => void;
};

type Props = {
  auth: Auth;
};

function AppRoutes({ auth }: Props) {
  return (
    <Routes>
      <Route
        path="/login"
        element={
          auth.isAuthed ? <Navigate to="/chat" replace /> : <LoginPage auth={auth} />
        }
      />

      <Route
        path="/chat/*"
        element={
          auth.isAuthed ? (
            <MainPage auth={auth} />
          ) : (
            <Navigate to="/login" replace />
          )
        }
      />

      <Route path="*" element={<Navigate to="/chat" replace />} />
    </Routes>
  );
}

export default AppRoutes;
