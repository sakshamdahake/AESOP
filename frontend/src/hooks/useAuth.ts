import { useState } from "react";

export function useAuth() {
  const [isAuthed, setIsAuthed] = useState(
    localStorage.getItem("aesop_auth") === "true"
  );

  const login = () => {
    localStorage.setItem("aesop_auth", "true");
    setIsAuthed(true);
  };

  const logout = () => {
    localStorage.removeItem("aesop_auth");
    setIsAuthed(false);
  };

  return { isAuthed, login, logout };
}
