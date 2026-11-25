importScripts("common.js");

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({ id: "fnds-selection", title: "Check with Fake News Detector", contexts: ["selection"] });
  chrome.contextMenus.create({ id: "fnds-page", title: "Check this page with Fake News Detector", contexts: ["page"] });
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  const payload = info.menuItemId === "fnds-selection" ? { text: info.selectionText } : { url: tab.url };
  await show(tab.id, { loading: true });
  try {
    await show(tab.id, { data: await checkWithServer(payload), server: await getServer() });
  } catch (e) {
    await show(tab.id, { error: e.message });
  }
});

function show(tabId, state) {
  return chrome.scripting.executeScript({ target: { tabId }, func: renderBox, args: [state] }).catch(() => {});
}

// Runs inside the web page: draws a small result box in the corner (built with DOM calls, no innerHTML).
function renderBox(state) {
  let box = document.getElementById("fnds-box");
  if (!box) {
    box = document.createElement("div");
    box.id = "fnds-box";
    Object.assign(box.style, { position: "fixed", right: "16px", bottom: "16px", zIndex: 2147483647, width: "320px",
      background: "#fff", color: "#1f2a37", border: "1px solid #d0d5dd", borderRadius: "10px", padding: "14px",
      boxShadow: "0 8px 24px rgba(0,0,0,.18)", font: "14px/1.45 system-ui, Segoe UI, Arial, sans-serif" });
    document.body.appendChild(box);
  }
  box.replaceChildren();
  const line = (text, style = {}) => { const d = document.createElement("div"); d.textContent = text; Object.assign(d.style, style); box.appendChild(d); return d; };
  const close = document.createElement("button");
  close.textContent = "×";
  Object.assign(close.style, { position: "absolute", top: "6px", right: "10px", border: "0", background: "none", fontSize: "18px", cursor: "pointer", color: "#667085" });
  close.onclick = () => box.remove();
  box.appendChild(close);
  line("Fake News Detector", { fontWeight: "600", color: "#1f3a5f", marginBottom: "6px" });
  if (state.loading) { line("Checking..."); return; }
  if (state.error) { line(state.error, { color: "#b42318" }); return; }
  const d = state.data, fake = d.label === "FAKE";
  line("Result: " + d.label + " (" + d.confidence + "%)", { fontSize: "18px", fontWeight: "700", color: fake ? "#b42318" : "#067647" });
  if (d.title) line(d.title, { fontWeight: "600", margin: "4px 0" });
  if (d.source) line("Source: " + d.source.domain + " - " + d.source.label, { fontSize: "12px", color: "#475467" });
  if (d.top_terms && d.top_terms.length) line("Key words: " + d.top_terms.join(", "), { fontSize: "13px", marginTop: "4px" });
  (d.signals.warnings || []).slice(0, 3).forEach(w => line("⚠ " + w, { fontSize: "12px", color: "#b54708" }));
  const a = document.createElement("a");
  a.href = state.server + "/history"; a.target = "_blank"; a.textContent = "Open history";
  Object.assign(a.style, { display: "inline-block", marginTop: "8px", fontSize: "12px", color: "#1f3a5f" });
  box.appendChild(a);
  line("A statistical prediction, not proof.", { fontSize: "11px", color: "#667085", marginTop: "6px" });
}
