import { useEffect, useState } from "react";

import ResultPage from "./pages/ResultPage";
import UploadPage from "./pages/UploadPage";

type Route =
  | { name: "upload" }
  | { name: "result"; jobId: string };

function readRoute(): Route {
  const match = window.location.pathname.match(/^\/result\/([^/]+)/);
  if (match) return { name: "result", jobId: match[1] };
  return { name: "upload" };
}

export default function App() {
  const [route, setRoute] = useState<Route>(() => readRoute());

  useEffect(() => {
    const onPop = () => setRoute(readRoute());
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  const navigateToResult = (jobId: string) => {
    window.history.pushState({}, "", `/result/${jobId}`);
    setRoute({ name: "result", jobId });
  };

  const navigateHome = () => {
    window.history.pushState({}, "", "/");
    setRoute({ name: "upload" });
  };

  return (
    <div className="app-shell">
      <header className="topbar">
        <button className="brand-button" type="button" onClick={navigateHome}>
          <span className="brand-mark">A</span>
          <span>
            <strong>AutoShotV2</strong>
            <small>Scene boundary analyzer</small>
          </span>
        </button>
      </header>

      {route.name === "upload" ? (
        <UploadPage onJobCreated={navigateToResult} />
      ) : (
        <ResultPage jobId={route.jobId} onBack={navigateHome} />
      )}
    </div>
  );
}
