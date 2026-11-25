const out = document.getElementById("out"), status = document.getElementById("status");

function render(d) {
  out.replaceChildren();
  const h = document.createElement("div");
  h.className = d.label; h.style.cssText = "font-size:18px;font-weight:700";
  h.textContent = "Result: " + d.label + " (" + d.confidence + "%)";
  out.appendChild(h);
  const add = (t) => { const p = document.createElement("div"); p.className = "muted"; p.textContent = t; out.appendChild(p); };
  if (d.title) add(d.title);
  if (d.source) add("Source: " + d.source.domain + " - " + d.source.label);
  if (d.top_terms.length) add("Key words: " + d.top_terms.join(", "));
  d.signals.warnings.slice(0, 3).forEach(w => add("⚠ " + w));
}

async function run(payload) {
  out.textContent = "Checking...";
  try { render(await checkWithServer(payload)); } catch (e) { out.textContent = e.message; }
}

document.getElementById("check").onclick = () => run({ text: document.getElementById("text").value });
document.getElementById("page").onclick = async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  run({ url: tab.url });
};

async function health() {
  const server = await getServer();
  document.getElementById("server").value = server;
  try {
    const d = await (await fetch(server + "/api/health")).json();
    status.textContent = "Connected: model " + d.model + " (" + d.algorithm + ")";
  } catch (e) { status.textContent = "Not connected to " + server; }
}

document.getElementById("save").onclick = async () => {
  const server = document.getElementById("server").value.trim().replace(/\/+$/, "") || DEFAULT_SERVER;
  const origin = new URL(server).origin + "/*";
  const granted = await chrome.permissions.request({ origins: [origin] }).catch(() => false);
  if (!granted && !server.startsWith("http://127.0.0.1:5000") && !server.startsWith("http://localhost:5000")) {
    status.textContent = "Permission for " + origin + " was not granted."; return;
  }
  await chrome.storage.sync.set({ server });
  health();
};
health();
