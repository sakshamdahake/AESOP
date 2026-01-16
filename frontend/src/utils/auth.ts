export const isAuthenticated = (): boolean => {
  return localStorage.getItem("aesop_auth") === "true";
};

export const login = () => {
  localStorage.setItem("aesop_auth", "true");
};

export const logout = () => {
  localStorage.removeItem("aesop_auth");
};
