type Props = {
  onLogout: () => void;
};

function TopBar({ onLogout }: Props) {
  return (
    <header
      style={{
        height: "56px",
        borderBottom: "1px solid #ddd",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "0 1rem",
      }}
    >
      <strong>AESOP</strong>
      <button onClick={onLogout}>Logout</button>
    </header>
  );
}

export default TopBar;
