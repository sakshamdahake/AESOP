import { useNavigate } from "react-router-dom";

type Props = {
  auth: {
    login: () => void;
  };
};

function LoginPage({ auth }: Props) {
  const navigate = useNavigate();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    auth.login();
    navigate("/");
  };

  return (
    <div style={{ padding: "2rem" }}>
      <h1>AESOP</h1>
      <p>Doctor Login</p>

      <form onSubmit={handleSubmit}>
        <input type="email" placeholder="Email" required />
        <br />
        <input type="password" placeholder="Password" required />
        <br />
        <button>Login</button>
      </form>
    </div>
  );
}

export default LoginPage;
