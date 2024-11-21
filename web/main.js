import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const style = `
.cpack-modal {
  position: fixed;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  background: #1a1a1a;
  padding: 20px;
  border-radius: 8px;
  z-index: 1000;
  color: white;
  min-width: 360px;
}

.cpack-input {
  width: 100%;
  padding: 8px;
  background: #333;
  border: 1px solid #444;
  border-radius: 4px;
  color: white;
  box-sizing: border-box;
}

.cpack-btn {
  padding: 6px 12px;
  background: #666;
  border: none;
  border-radius: 4px;
  color: white;
  cursor: pointer;
}

.cpack-btn.primary {
  background: #00a67d;
}

.cpack-btn-container {
  margin-top: 20px;
  display: flex;
  justify-content: flex-end;
  gap: 10px;
}

.cpack-title {
  margin-bottom: 15px;
  font-size: 1.3em;
  font-weight: bold;
}

.cpack-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.5);
  z-index: 999;
}

.cpack-input-row {
  display: flex;
  align-items: center;
  gap: 5px;
  margin-bottom: 5px;
}

.cpack-form-item {
  margin-bottom: 15px;
}

.cpack-form-item label {
  margin-bottom: 5px;
}

#build-info {
  display: flex;
  padding: 10px;
  flex-direction: column;
  align-items: center;
}

#build-error {
  display: none;
  color: #ff8383;
  text-wrap: auto;
  overflow-x: auto;
  background: #333;
  padding: 10px;
}
`

function createModal(modal) {
  const overlay = document.createElement("div");
  overlay.className = "cpack-overlay";

  document.body.appendChild(overlay);
  document.body.appendChild(modal);

  return {
    close: () => {
      modal.remove();
      overlay.remove();
    }
  };
}

function createInputModal() {
  return new Promise((resolve) => {
    const modal = document.createElement("div");
    modal.className = "cpack-modal";

    const title = document.createElement("div");
    title.textContent = "Package Worlflow";
    title.className = "cpack-title";

    const input = document.createElement("input");
    input.type = "text";
    input.value = "package";
    input.className = "cpack-input";

    const buttonContainer = document.createElement("div");
    buttonContainer.className = "cpack-btn-container";

    const confirmButton = document.createElement("button");
    confirmButton.textContent = "Confirm";
    confirmButton.className = "cpack-btn primary";

    const cancelButton = document.createElement("button");
    cancelButton.textContent = "Cancel";
    cancelButton.className = "cpack-btn";

    buttonContainer.appendChild(cancelButton);
    buttonContainer.appendChild(confirmButton);
    modal.appendChild(title);
    modal.appendChild(input);
    modal.appendChild(buttonContainer);

    const { close } = createModal(modal);

    confirmButton.onclick = () => {
      const filename = input.value.trim();
      if (filename) {
        close();
        resolve(filename);
      }
    };

    cancelButton.onclick = () => {
      close();
      resolve(null);
    };

    input.addEventListener("keyup", (e) => {
      if (e.key === "Enter") {
        confirmButton.click();
      }
    });

    input.select();
  });
}

function createDownloadModal() {
  const modal = document.createElement("div");
  modal.className = "cpack-modal";

  const title = document.createElement("div");
  title.textContent = "Packaging...";
  title.style.marginBottom = "15px";
  title.style.color = "#fff";

  const progress = document.createElement("div");
  progress.style.cssText = `
    width: 100%;
    height: 20px;
    background: #333;
    border-radius: 10px;
    overflow: hidden;
  `;

  const progressBar = document.createElement("div");
  progressBar.style.cssText = `
    width: 0%;
    height: 100%;
    background: #00a67d;
    transition: width 0.3s ease;
  `;

  progress.appendChild(progressBar);
  modal.appendChild(title);
  modal.appendChild(progress);

  const { close } = createModal(modal);

  return {
    updateProgress: (percent) => {
      progressBar.style.width = `${percent}%`;
    },
    close
  };
}

async function downloadPackage(event) {
  const filename = await createInputModal();
  if (!filename) return;

  const button = event.target;

  button.disabled = true;
  const downloadModal = createDownloadModal();

  try {
    downloadModal.updateProgress(20);
    const { workflow, output: workflow_api } = await app.graphToPrompt();

    downloadModal.updateProgress(40);
    const body = JSON.stringify({ workflow, workflow_api });

    downloadModal.updateProgress(60);
    const resp = await api.fetchApi("/bentoml/pack", { method: "POST", body, headers: { "Content-Type": "application/json" } });

    downloadModal.updateProgress(80);
    const downloadUrl = (await resp.json())["download_url"];

    downloadModal.updateProgress(100);
    const link = document.createElement("a");
    link.href = downloadUrl;
    link.download = filename + ".cpack.zip";
    link.click();

    setTimeout(() => {
      downloadModal.close();
    }, 1000);
  } catch (error) {
    console.error("Package failed:", error);
    downloadModal.close();
  } finally {
    button.disabled = false;
  }
}

const buildForm = `
<div class="cpack-form-item">
  <label for="bentoName">Bento Name</label>
  <input type="text" class="cpack-input" name="bentoName" value="comfy-pack" />
</div>
<div class="cpack-form-item">
  <label for="systemPackages">System Packages</label>
  <div id="system-packages-array">
    <button class="cpack-btn" id="add-button" style="margin: 5px 0px">Add</button>
  </div>
</div>
<div class="cpack-form-item">
  <label for="push">Push to Cloud</label>
  <input type="checkbox" name="push" id="push-switch" />
</div>
<div id="credentials-group" style="display: none">
  <div class="cpack-form-item">
    <label for="bentoName">BentoCloud API Key</label>
    <input type="password" class="cpack-input" name="apiKey" />
  </div>
  <div class="cpack-form-item">
    <label for="bentoName">BentoCloud Endpoint</label>
    <input type="text" class="cpack-input" name="endpoint" />
  </div>
  <p style="font-size: 0.85em;">Leave these empty to use the credentials stored in local machine</p>
</div>
`

function createBuildModal() {
  const modal = document.createElement("div");
  modal.className = "cpack-modal";
  modal.style.width = "400px";

  const title = document.createElement("div");
  title.textContent = "Build Workflow As Bento";
  title.className = "cpack-title";

  const form = document.createElement("form");
  form.innerHTML = buildForm;

  const addButton = form.querySelector("#add-button");
  const systemPackagesArray = form.querySelector("#system-packages-array");
  addButton.addEventListener("click", (e) => {
    e.preventDefault();
    const row = document.createElement("div");
    row.className = "cpack-input-row";
    row.innerHTML = `
      <div style="flex: 1"><input type="text" class="cpack-input" name="systemPackages" /></div>
      <button class="cpack-btn" style="margin-left: 10px">Remove</button>
    `
    systemPackagesArray.appendChild(row);
    row.querySelector("button").onclick = (e) => {
      e.preventDefault();
      row.remove();
    }
  });

  const pushSwitch = form.querySelector("#push-switch");
  const credentialsGroup = form.querySelector("#credentials-group");
  pushSwitch.addEventListener("change", (e) => {
    credentialsGroup.style.display = e.target.checked ? "block" : "none";
  });

  const buttonContainer = document.createElement("div");
  buttonContainer.className = "cpack-btn-container";

  const confirmButton = document.createElement("button");
  confirmButton.textContent = "Build";
  confirmButton.className = "cpack-btn primary";

  const cancelButton = document.createElement("button");
  cancelButton.textContent = "Cancel";
  cancelButton.className = "cpack-btn";

  buttonContainer.appendChild(cancelButton);
  buttonContainer.appendChild(confirmButton);

  modal.appendChild(title);
  modal.appendChild(form);
  modal.appendChild(buttonContainer);

  const { close } = createModal(modal);
  cancelButton.onclick = close;
  return new Promise((resolve) => {
    form.querySelector("input[name='bentoName']").select();
    confirmButton.onclick = async (e) => {
      e.preventDefault();
      const formData = new FormData(form);
      const { workflow, output: workflow_api } = await app.graphToPrompt();
      const data = {
        bento_name: formData.get("bentoName"),
        system_packages: Array.from(formData.getAll("systemPackages").filter(Boolean)),
        push: formData.get("push") === "on",
        api_key: formData.get("apiKey"),
        endpoint: formData.get("endpoint"),
        workflow,
        workflow_api
      };
      close();
      resolve(data);
    }
  });
}

const spinner = `<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24">
  <path fill="currentColor" d="M12,1A11,11,0,1,0,23,12,11,11,0,0,0,12,1Zm0,19a8,8,0,1,1,8-8A8,8,0,0,1,12,20Z" opacity=".25"/>
  <path fill="currentColor" d="M10.14,1.16a11,11,0,0,0-9,8.92A1.59,1.59,0,0,0,2.46,12,1.52,1.52,0,0,0,4.11,10.7a8,8,0,0,1,6.66-6.61A1.42,1.42,0,0,0,12,2.69h0A1.57,1.57,0,0,0,10.14,1.16Z">
    <animateTransform attributeName="transform" dur="0.75s" repeatCount="indefinite" type="rotate" values="0 12 12;360 12 12"/>
  </path>
</svg>`

async function createBuildingModal(data) {
  const modal = document.createElement("div");
  modal.className = "cpack-modal";
  modal.style.width = "400px";

  const title = document.createElement("div");
  title.textContent = "Building...";
  title.className = "cpack-title";

  const body = document.createElement("div");
  body.innerHTML = `
    <div id="build-info"></div>
    <pre id="build-error"></pre>
  `;

  const buttonContainer = document.createElement("div");
  buttonContainer.className = "cpack-btn-container";

  const closeButton = document.createElement("button");
  closeButton.textContent = "Close";
  closeButton.className = "cpack-btn";
  buttonContainer.appendChild(closeButton);

  modal.appendChild(title);
  modal.appendChild(body);
  modal.appendChild(buttonContainer);

  const { close } = createModal(modal);
  closeButton.onclick = close;
  closeButton.disabled = true;

  const info = body.querySelector("#build-info");
  const setError = (error) => {
    title.textContent = "Build Failed";
    info.style.display = "none";
    const pre = body.querySelector("#build-error");
    pre.textContent = error;
    pre.style.display = "block";
  }
  info.innerHTML = spinner;
  try {
    const resp = await api.fetchApi("/bentoml/build", {
      method: "POST",
      body: JSON.stringify(data),
      headers: { "Content-Type": "application/json" }
    });
    const respData = await resp.json();
    if (respData.error) {
      setError(respData.error);
    } else {
      info.innerHTML = `<p><strong>${respData.bento}</strong></p>`;
      title.textContent = "Build Completed";
    }
  } catch(e) {
    setError(e.message);
  } finally {
    closeButton.disabled = false;
  }
}

app.registerExtension({
  name: "Comfy.CPackExtension",

  async setup() {
    const styleTag = document.createElement("style");
    styleTag.innerHTML = style;
    document.head.appendChild(styleTag);
    const menu = document.querySelector(".comfy-menu");
    const separator = document.createElement("hr");

    separator.style.margin = "20px 0";
    separator.style.width = "100%";
    menu.append(separator);
    const packButton = document.createElement("button");
    packButton.textContent = "Package";
    packButton.onclick = downloadPackage;
    menu.append(packButton);

    const buildButton = document.createElement("button");
    buildButton.textContent = "Build";
    buildButton.onclick = async () => {
      const data = await createBuildModal();
      console.log(data);
      await createBuildingModal(data);
    };
    menu.append(buildButton);
  }
});
