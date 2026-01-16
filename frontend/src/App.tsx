import AppRoutes from "./routes/AppRoutes";
import { useAuth } from "./hooks/useAuth";

function App() {
  const auth = useAuth();
  return <AppRoutes auth={auth} />;
}

export default App;
