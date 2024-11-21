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
  min-width: 300px;
}

.cpack-input {
  width: 100%;
  padding: 8px;
  margin-bottom: 15px;
  background: #333;
  border: 1px solid #444;
  border-radius: 4px;
  color: #fff;
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
  display: flex;
  justify-content: flex-end;
  gap: 10px;
}

.cpack-title {
  margin-bottom: 15px;
  color: #fff;
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

    const title = document.createElement("h3");
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

  const title = document.createElement("h3");
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

function buildBento() {

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
    buildButton.onclick = buildBento;
    menu.append(buildButton);
  }
});
