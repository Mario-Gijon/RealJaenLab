/* Supervisor library and resumable managed-original importer. */
(() => {
  const root = document.querySelector("#library-screen");
  const CHUNK_BYTES = 16 * 1024 * 1024;
  const KIND_LABELS = {
    full_match: "Partido completo", first_half: "1ª Parte", second_half: "2ª Parte",
    clip: "Clip", prematch: "Prepartido", other: "Vídeo",
  };
  const state = { catalog: null, file: null, upload: null, localFilename: null, uploadStartedAt: 0, uploadStartOffset: 0 };

  const escapeHtml = (value) => String(value ?? "").replace(/[&<>'"]/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", "\"": "&quot;" }[character]));
  const formatBytes = (bytes) => {
    const units = ["B", "KB", "MB", "GB", "TB"]; let value = Number(bytes) || 0; let unit = 0;
    while (value >= 1024 && unit < units.length - 1) { value /= 1024; unit += 1; }
    return `${value >= 10 || unit === 0 ? value.toFixed(0) : value.toFixed(2)} ${units[unit]}`;
  };
  const api = async (url, options) => {
    const response = await fetch(url, options);
    if (!response.ok) {
      let detail = "";
      try { detail = (await response.json()).detail; } catch { /* concise client error below */ }
      throw new Error(detail || "No se pudo completar la operación.");
    }
    return response.json();
  };
  const currentAsset = (asset) => ({
    assetId: asset.id,
    // Local input files use their filename; managed originals use their asset id.
    name: asset.storage_type === "local_input"
      ? String(asset.relative_path || "").split("/").pop()
      : asset.id,
    displayName: asset.display_name,
    playbackUrl: `/api/library/assets/${encodeURIComponent(asset.id)}/video`,
    ...(asset.technical || {}),
  });

  function assetsForMatch(matchId) { return state.catalog.assets.filter((asset) => asset.match_id === matchId); }
  function venueForMatch(match) { return state.catalog.venues.find((venue) => venue.id === match.venue_id); }
  function assetRow(asset, { local = false } = {}) {
    const name = asset.display_name;
    const action = local
      ? `<button class="button button-secondary" data-organize="${escapeHtml(asset.filename)}" type="button">Organizar</button>`
      : `<div class="asset-actions"><button class="button button-secondary" data-open="${escapeHtml(asset.id)}" type="button">Abrir</button><button class="button button-secondary" data-edit="${escapeHtml(asset.id)}" type="button">Editar</button></div>`;
    return `<div class="asset-row"><div><strong>${escapeHtml(name)}</strong><div class="asset-meta">${escapeHtml(KIND_LABELS[asset.kind] || "Vídeo")}</div></div>${action}</div>`;
  }
  function render() {
    const catalog = state.catalog;
    if (!catalog) { root.innerHTML = ""; return; }
    const matches = catalog.matches || [];
    const organized = matches.map((match) => {
      const venue = venueForMatch(match);
      const assets = assetsForMatch(match.id);
      return `<article class="match-card"><h3>${escapeHtml(match.title)}</h3><div class="match-meta">${escapeHtml(venue?.name || match.date || "Sin estadio ni fecha")}</div><div class="match-meta">${assets.length} vídeo${assets.length === 1 ? "" : "s"}</div><div class="match-assets">${assets.map((asset) => assetRow(asset)).join("") || "<span class=\"library-note\">Todavía no hay vídeos.</span>"}</div></article>`;
    }).join("");
    const unorganized = (catalog.assets || []).filter((asset) => !asset.match_id);
    const pending = (state.pendingUploads || []).filter((upload) => upload.status === "uploading");
    root.innerHTML = `<div class="library-shell">
      <header class="library-header"><div><span class="library-kicker">REALJAÉN SUPERVISOR</span><h1 class="library-title">Tus partidos</h1></div><button class="button button-primary" id="import-video" type="button">+ Importar vídeo</button></header>
      ${pending.length ? `<section class="library-section"><div class="library-empty">Hay una subida pendiente: ${escapeHtml(pending[0].original_filename)} · ${Math.round((pending[0].next_offset / pending[0].total_size_bytes) * 100)}% <button class="button button-secondary" data-resume="${escapeHtml(pending[0].upload_id)}" type="button">Seleccionar archivo para continuar</button> <button class="button button-secondary" data-cancel-upload="${escapeHtml(pending[0].upload_id)}" type="button">Cancelar subida</button></div></section>` : ""}
      <section class="library-section">${matches.length ? `<div class="match-grid">${organized}</div>` : `<div class="library-empty">Todavía no hay partidos.<br><br><button class="button button-primary" id="import-empty" type="button">Importar vídeo</button></div>`}</section>
      ${unorganized.length ? `<section class="library-section"><h2>Sin organizar</h2><div class="asset-list">${unorganized.map((asset) => assetRow(asset)).join("")}</div></section>` : ""}
      ${(catalog.local_unregistered || []).length ? `<section class="library-section"><h2>Vídeos locales sin organizar</h2><div class="asset-list">${catalog.local_unregistered.map((asset) => assetRow(asset, { local: true })).join("")}</div></section>` : ""}
    </div>`;
    root.querySelector("#import-video")?.addEventListener("click", () => openImport());
    root.querySelector("#import-empty")?.addEventListener("click", () => openImport());
    root.querySelectorAll("[data-open]").forEach((button) => button.addEventListener("click", () => {
      const asset = catalog.assets.find((item) => item.id === button.dataset.open);
      if (asset) window.SupervisorWorkspace.openVideo(currentAsset(asset));
    }));
    root.querySelectorAll("[data-edit]").forEach((button) => button.addEventListener("click", () => {
      const asset = catalog.assets.find((item) => item.id === button.dataset.edit);
      if (asset) openEdit(asset);
    }));
    root.querySelectorAll("[data-organize]").forEach((button) => button.addEventListener("click", () => openImport(button.dataset.organize)));
    root.querySelectorAll("[data-resume]").forEach((button) => button.addEventListener("click", () => openPending(button.dataset.resume)));
    root.querySelectorAll("[data-cancel-upload]").forEach((button) => button.addEventListener("click", () => cancelUpload(button.dataset.cancelUpload)));
  }
  async function refresh() {
    try {
      const [catalog, uploads] = await Promise.all([api("/api/library"), api("/api/library/uploads")]);
      state.catalog = catalog; state.pendingUploads = uploads.uploads || [];
      render();
    } catch { root.innerHTML = `<div class="library-shell"><div class="library-empty">No se pudo cargar la biblioteca.</div></div>`; }
  }
  function options(items, label) { return `<option value="">${label}</option>${items.map((item) => `<option value="${escapeHtml(item.id)}">${escapeHtml(item.title || item.name)}</option>`).join("")}`; }
  function dialogHtml() {
    const catalog = state.catalog;
    const local = Boolean(state.localFilename);
    return `<div class="library-modal" role="dialog" aria-modal="true" aria-labelledby="import-title"><div class="import-dialog">
      <div class="import-dialog-header"><h2 id="import-title">${local ? "Organizar vídeo local" : "Importar vídeo"}</h2><button class="import-close" id="import-close" type="button" aria-label="Cerrar">×</button></div>
      ${local ? `<div class="library-note">${escapeHtml(state.localFilename)} · el archivo original no se moverá.</div>` : `<div class="drop-zone" id="drop-zone"><div><strong>Arrastra aquí tu vídeo</strong><p>o</p><button class="button button-secondary" id="choose-file" type="button">Seleccionar archivo</button><input id="file-input" type="file" accept="video/mp4" /></div></div><div class="library-note" id="file-info">Solo MP4 · Original · sin pérdida de calidad</div>`}
      <form class="import-form" id="import-form"><label>Nombre<input id="asset-name" required value="${escapeHtml(local ? state.localFilename.replace(/\.mp4$/i, "") : "")}" /></label><label>Tipo<select id="asset-kind">${Object.entries(KIND_LABELS).map(([value, label]) => `<option value="${value}">${label}</option>`).join("")}</select></label>
      <label>Partido<select id="match-select">${options(catalog.matches, "Sin organizar")}</select></label><button class="button button-secondary" id="new-match-toggle" type="button">+ Crear nuevo partido</button>
      <div class="import-inline" id="new-match" hidden><label>Partido<input id="match-title" /></label><label>Fecha <input id="match-date" type="date" /></label><label>Estadio<select id="venue-select">${options(catalog.venues, "Sin estadio")}</select></label><button class="button button-secondary" id="new-venue-toggle" type="button">+ Crear estadio</button><div id="new-venue" hidden><label>Nombre del estadio<input id="venue-name" /></label></div></div>
      <div class="import-error" id="import-error" hidden></div><div class="upload-progress" id="upload-progress" hidden></div><div class="import-actions"><button class="button button-secondary" id="cancel-import" type="button">Cancelar</button><button class="button button-primary" id="submit-import" type="submit">${local ? "Organizar vídeo" : "Subir vídeo"}</button></div></form>
    </div></div>`;
  }
  function openImport(localFilename = null) {
    state.file = null; state.upload = null; state.resumeExisting = false; state.localFilename = localFilename;
    document.body.insertAdjacentHTML("beforeend", dialogHtml());
    const modal = document.body.lastElementChild; const close = () => modal.remove();
    const input = modal.querySelector("#file-input"); const info = modal.querySelector("#file-info"); const zone = modal.querySelector("#drop-zone");
    const choose = () => input?.click();
    modal.querySelector("#choose-file")?.addEventListener("click", choose);
    modal.querySelector("#import-close").addEventListener("click", close); modal.querySelector("#cancel-import").addEventListener("click", close);
    modal.querySelector("#new-match-toggle").addEventListener("click", () => { modal.querySelector("#new-match").hidden = !modal.querySelector("#new-match").hidden; });
    modal.querySelector("#new-venue-toggle").addEventListener("click", () => { modal.querySelector("#new-venue").hidden = !modal.querySelector("#new-venue").hidden; });
    const selectFile = (file) => { if (!file) return; if (!file.name.toLowerCase().endsWith(".mp4")) return setError(modal, "Selecciona un archivo MP4."); state.file = file; info.textContent = `${file.name} · ${formatBytes(file.size)} · Original · sin pérdida de calidad`; modal.querySelector("#asset-name").value ||= file.name.replace(/\.mp4$/i, ""); };
    input?.addEventListener("change", () => selectFile(input.files[0]));
    ["dragenter", "dragover"].forEach((name) => zone?.addEventListener(name, (event) => { event.preventDefault(); zone.classList.add("is-dragging"); }));
    ["dragleave", "drop"].forEach((name) => zone?.addEventListener(name, (event) => { event.preventDefault(); zone.classList.remove("is-dragging"); }));
    zone?.addEventListener("drop", (event) => selectFile(event.dataTransfer.files[0]));
    modal.querySelector("#import-form").addEventListener("submit", async (event) => { event.preventDefault(); await submitImport(modal, close); });
  }
  function editDialogHtml(asset) {
    const catalog = state.catalog;
    return `<div class="library-modal" role="dialog" aria-modal="true" aria-labelledby="edit-title"><div class="import-dialog">
      <div class="import-dialog-header"><h2 id="edit-title">Editar vídeo</h2><button class="import-close" id="edit-close" type="button" aria-label="Cerrar">×</button></div>
      <form class="import-form" id="edit-form"><label>Nombre<input id="asset-name" required value="${escapeHtml(asset.display_name)}" /></label><label>Tipo<select id="asset-kind">${Object.entries(KIND_LABELS).map(([value, label]) => `<option value="${value}"${asset.kind === value ? " selected" : ""}>${label}</option>`).join("")}</select></label>
      <label>Partido<select id="match-select">${options(catalog.matches, "Sin organizar")}</select></label><button class="button button-secondary" id="new-match-toggle" type="button">+ Crear nuevo partido</button>
      <div class="import-inline" id="new-match" hidden><label>Partido<input id="match-title" /></label><label>Fecha <input id="match-date" type="date" /></label><label>Estadio<select id="venue-select">${options(catalog.venues, "Sin estadio")}</select></label><button class="button button-secondary" id="new-venue-toggle" type="button">+ Crear estadio</button><div id="new-venue" hidden><label>Nombre del estadio<input id="venue-name" /></label></div></div>
      <div class="import-error" id="import-error" hidden></div><div class="import-actions"><button class="button button-secondary" id="cancel-edit" type="button">Cancelar</button><button class="button button-primary" type="submit">Guardar cambios</button></div></form>
    </div></div>`;
  }
  function openEdit(asset) {
    document.body.insertAdjacentHTML("beforeend", editDialogHtml(asset));
    const modal = document.body.lastElementChild;
    modal.querySelector("#match-select").value = asset.match_id || "";
    const close = () => modal.remove();
    modal.querySelector("#edit-close").addEventListener("click", close);
    modal.querySelector("#cancel-edit").addEventListener("click", close);
    modal.querySelector("#new-match-toggle").addEventListener("click", () => { modal.querySelector("#new-match").hidden = !modal.querySelector("#new-match").hidden; });
    modal.querySelector("#new-venue-toggle").addEventListener("click", () => { modal.querySelector("#new-venue").hidden = !modal.querySelector("#new-venue").hidden; });
    modal.querySelector("#edit-form").addEventListener("submit", async (event) => {
      event.preventDefault();
      try {
        const display_name = modal.querySelector("#asset-name").value.trim();
        if (!display_name) throw new Error("Indica un nombre para el vídeo.");
        const relationships = await relationshipPayload(modal);
        await api(`/api/library/assets/${encodeURIComponent(asset.id)}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ display_name, kind: modal.querySelector("#asset-kind").value, match_id: relationships.match_id }),
        });
        close();
        await refresh();
      } catch (error) { setError(modal, error.message || "No se pudo guardar el vídeo."); }
    });
  }
  function setError(modal, message) { const node = modal.querySelector("#import-error"); node.textContent = message; node.hidden = false; }
  async function relationshipPayload(modal) {
    let matchId = modal.querySelector("#match-select").value || null;
    if (!modal.querySelector("#new-match").hidden) {
      let venueId = modal.querySelector("#venue-select").value || null;
      if (!modal.querySelector("#new-venue").hidden) {
        const name = modal.querySelector("#venue-name").value.trim(); if (!name) throw new Error("Indica el nombre del estadio.");
        venueId = (await api("/api/library/venues", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name }) })).venue.id;
      }
      const title = modal.querySelector("#match-title").value.trim(); if (!title) throw new Error("Indica el nombre del partido.");
      matchId = (await api("/api/library/matches", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ title, date: modal.querySelector("#match-date").value || null, venue_id: venueId }) })).match.id;
    }
    return { match_id: matchId };
  }
  function progress(modal, session) {
    const transferred = Number(session.next_offset || 0); const total = Number(session.total_size_bytes || 1); const elapsed = Math.max((Date.now() - state.uploadStartedAt) / 1000, .1); const rate = Math.max(0, (transferred - state.uploadStartOffset) / elapsed); const remaining = rate ? Math.ceil((total - transferred) / rate) : null;
    const node = modal.querySelector("#upload-progress"); node.hidden = false; node.innerHTML = `<strong>Subiendo vídeo…</strong><span>${formatBytes(transferred)} / ${formatBytes(total)} · ${Math.round((transferred / total) * 100)}%</span><div class="upload-meter"><span style="width:${(transferred / total) * 100}%"></span></div><span class="upload-copy">${rate ? `${formatBytes(rate)}/s${remaining ? ` · unos ${remaining}s restantes` : ""}` : "Preparando subida…"}</span><span class="upload-copy">Original · sin pérdida de calidad</span><button class="button button-secondary" id="cancel-upload-now" type="button">Cancelar</button>`;
    node.querySelector("#cancel-upload-now").addEventListener("click", () => cancelUpload(state.upload.upload_id, modal));
  }
  async function submitImport(modal, close) {
    try {
      if (state.resumeExisting) {
        if (!state.file || state.file.name !== state.upload.original_filename || state.file.size !== state.upload.total_size_bytes) {
          throw new Error("El archivo seleccionado no coincide con la subida pendiente.");
        }
        await uploadSequential(modal, close);
        return;
      }
      const relationships = await relationshipPayload(modal); const name = modal.querySelector("#asset-name").value.trim(); const kind = modal.querySelector("#asset-kind").value;
      if (!name) throw new Error("Indica un nombre para el vídeo.");
      if (state.localFilename) {
        await api("/api/library/assets/register", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ input_filename: state.localFilename, display_name: name, kind, ...relationships }) });
        close(); await refresh(); return;
      }
      if (!state.file) throw new Error("Selecciona un archivo MP4.");
      state.upload = await api("/api/library/uploads", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ original_filename: state.file.name, total_size_bytes: state.file.size, display_name: name, kind, ...relationships }) });
      localStorage.setItem("supervisor-upload", JSON.stringify({ upload_id: state.upload.upload_id, original_filename: state.file.name, total_size_bytes: state.file.size, last_modified: state.file.lastModified }));
      await uploadSequential(modal, close);
    } catch (error) { setError(modal, error.message || "No se pudo iniciar la subida."); }
  }
  async function uploadSequential(modal, close) {
    state.upload = await api(`/api/library/uploads/${encodeURIComponent(state.upload.upload_id)}`); state.uploadStartedAt = Date.now(); state.uploadStartOffset = state.upload.next_offset;
    try {
      while (state.upload.next_offset < state.upload.total_size_bytes) {
        const start = state.upload.next_offset; const endExclusive = Math.min(start + CHUNK_BYTES, state.upload.total_size_bytes); const response = await fetch(`/api/library/uploads/${encodeURIComponent(state.upload.upload_id)}/chunk`, { method: "PUT", headers: { "Content-Range": `bytes ${start}-${endExclusive - 1}/${state.upload.total_size_bytes}` }, body: state.file.slice(start, endExclusive) });
        if (!response.ok) throw new Error("La conexión se ha interrumpido.");
        state.upload = await response.json(); progress(modal, state.upload);
      }
      const completed = await api(`/api/library/uploads/${encodeURIComponent(state.upload.upload_id)}/complete`, { method: "POST" });
      localStorage.removeItem("supervisor-upload"); modal.querySelector("#upload-progress").innerHTML = `<span class="import-success">Vídeo importado ✓</span>`; await refresh(); setTimeout(close, 650); return completed.asset;
    } catch (error) { try { state.upload = await api(`/api/library/uploads/${encodeURIComponent(state.upload.upload_id)}`); } catch { /* server session may be unavailable */ } progress(modal, state.upload); setError(modal, "Se ha interrumpido la conexión."); const resume = document.createElement("button"); resume.className = "button button-primary"; resume.type = "button"; resume.textContent = "Reanudar"; resume.addEventListener("click", () => uploadSequential(modal, close)); modal.querySelector("#upload-progress").append(resume); }
  }
  async function openPending(uploadId) {
    state.localFilename = null; openImport(); const modal = document.body.lastElementChild; const session = await api(`/api/library/uploads/${encodeURIComponent(uploadId)}`); state.upload = session;
    state.resumeExisting = true;
    modal.querySelector("#asset-name").value = session.display_name || "";
    modal.querySelector("#asset-kind").value = session.kind || "other";
    modal.querySelector("#submit-import").textContent = "Reanudar subida";
  }
  async function cancelUpload(uploadId, modal = null) { if (!window.confirm("¿Cancelar esta subida? Se descartarán los bytes transferidos.")) return; try { await api(`/api/library/uploads/${encodeURIComponent(uploadId)}`, { method: "DELETE" }); localStorage.removeItem("supervisor-upload"); modal?.remove(); await refresh(); } catch (error) { if (modal) setError(modal, "No se pudo cancelar la subida."); } }
  window.SupervisorLibrary = { show: refresh };
  refresh();
})();
