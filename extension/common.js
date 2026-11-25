// Shared helpers for the popup and the background worker.
const DEFAULT_SERVER = "http://127.0.0.1:5000";

async function getServer() {
  const { server } = await chrome.storage.sync.get({ server: DEFAULT_SERVER });
  return server.replace(/\/+$/, "");
}

async function checkWithServer(payload) {
  const server = await getServer();
  let r;
  try {
    r = await fetch(server + "/api/predict", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
  } catch (e) {
    throw new Error("Cannot reach " + server + ". Is app.py running?");
  }
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || ("Server error " + r.status));
  return data;
}
