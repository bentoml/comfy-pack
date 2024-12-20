# Comfy-Pack: Making ComfyUI Workflows Shareable

![banner2](https://github.com/user-attachments/assets/14a7e469-6683-4818-9d54-5e5a8d0aa454)


comfy-pack is a comprehensive toolkit for reliably packing and unpacking environments for ComfyUI workflows. 


- 📦 **Pack workflow environments as artifacts:** Saves the workflow environment in a `.cpack.zip` artifact with Python package versions, ComfyUI and custom node revisions, and model hashes.
- ✨ **Unpack artifacts to recreate workflow environments:** Unpacks the `.cpack.zip` artifact to recreate the same environment with the exact Python package versions, ComfyUI and custom node revisions, and model weights.
- 🚀 **Deploy workflows as APIs:** Deploys the workflow as a RESTful API with customizable input and output parameters.

## Motivations
ComfyUI Manager is great for find missing custom nodes. But when sharing ComfyUI workflows to others(your audience or team members), you've still likely heard these responses:

- "Custom Node not found"
- "Cannot find the correct model file"
- "Missing Python dependencies"

These are fundamental challenges in workflow sharing – every component should match exactly: custom nodes, model files, and Python dependencies. Modern pacakge managers like npm and poetry introduced "lock" feature, which means record the exact version for every requirement. ComfyUI Manager isn't designed for that.

We learned it from our community and developed comfy-pack to address these problems. With a single click, it captures and locks your entire workflow environment into a `.cpack.zip` file, including Python packages, custom nodes, model hashes, and required assets.

Users can recreate the exact environment with one command:

```bash
comfy-pack unpack workflow.cpack.zip
```

This means you can focus on your creative work while comfy-pack handles the rest.

## Usages

### Installation

We recommend you use ComfyUI Manager to install comfy-pack. Simply search for `comfy-pack` and click **Install**. Restart the server and refresh your ComfyUI interface to apply changes.

![install_node](https://github.com/user-attachments/assets/dbfb730d-edff-4a52-b6c4-695e3ec70368)

Alternatively, clone the project repository through `git`.

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/bentoml/comfy-pack.git
```

To install the comfy-pack CLI, run:

```bash
pip install comfy-pack
```

### Pack a ComfyUI workflow and its environment

You can package a workflow and the environment required to run the workflow into an artifact that can be unpacked elsewhere.

1. Click the **Package** button to create a `.cpack.zip` artifact.
2. (Optional) Select the models that you want to include (only model hash will be recorded, so you won't get a 100GB zip file).

![pack](https://github.com/user-attachments/assets/e08bbed2-84dc-474e-a701-6c6db16edf76)

### Unpack the ComfyUI environments

Unpacking a `.cpack.zip` artifact will restore the ComfyUI environment for the workflow. During unpacking, comfy-pack will perform the following steps.

1. Prepare a Python virtual environment with the exact packages used to run the workflow.
2. Clone ComfyUI and custom nodes from the exact revisions required by the workflow.
3. Search for and download models from common registries like Hugging Face and Civitai. Unpacking workflows using the same model will not cause the model to be downloaded multiple times. Instead, model weights will be symbolically linked.

To unpack:

```bash
comfy-pack unpack workflow.cpack.zip
```

For example cpack files, check our [examples folder](examples/).

### Deploy a workflow as an API

You can turn a ComfyUI workflow into an API endpoint callable using any clients through HTTP.

<details>
<summary> 1. Annotate input & output </summary>

Use custom nodes provided by comfy-pack to annotate the fields to be used as input and output parameters. To add a comfy-pack node, right-click and select **Add Node** > **ComfyPack** > **output/input** > [Select a type]

Input nodes:

- ImageInput: Accepts `image` type input, similar to the official `LoadImage` node
- StringInput: Accepts `string` type input (e.g., prompts)
- IntInput: Accepts `int` type input (e.g., dimensions, seeds)
- AnyInput: Accepts `combo` type and more input (e.g., custom nodes)

![input](https://github.com/user-attachments/assets/44264007-0ac8-4e23-8dc0-e60aa0ebcea2)

Output nodes:

- ImageOutput: Outputs `image` type, similar to the official `SaveImage` node
- FileOutput: Outputs file path as `string` type and saves the file under that path

![output](https://github.com/user-attachments/assets/a4526661-8930-4575-bacc-33b6887f6271)

More field types are under way.
</details>

<details>
<summary> 2. Serve the workflow </summary>

Start an HTTP server at `http://127.0.0.1:3000` (default) to serve the workflow under the `/generate` path.

![serve](https://github.com/user-attachments/assets/8d4c92c5-d6d7-485e-bc71-e4fc0fe8bf35)

You can call the `/generate` endpoint by specifying parameters configured through your comfy-pack nodes, such as prompt, width, height, and seed.

> [!NOTE]
> The name of a comfy-pack node is the parameter name used for API calls.

Examples to call the endpoint:

CURL

```bash
curl -X 'POST' \
  'http://127.0.0.1:3000/generate' \
  -H 'accept: application/octet-stream' \
  -H 'Content-Type: application/json' \
  -d '{
  "prompt": "rocks in a bottle",
  "width": 512, 
  "height": 512,
  "seed": 1
}'
```

BentoML client

Under the hood, comfy-pack leverages [BentoML](https://github.com/bentoml/BentoML), the unified model serving framework. You can invoke the endpoint using [the BentoML Python client](https://docs.bentoml.com/en/latest/build-with-bentoml/clients.html):

```python
import bentoml

with bentoml.SyncHTTPClient("http://127.0.0.1:3000") as client:
        result = client.generate(
            prompt="rocks in a bottle",
            width=512,
            height=512,
            seed=1
        )
```

</details>

<details>
<summary> 3. (Optional) Pack the workflow and environment </summary>

Pack the workflow and environment into an artifact that can be unpacked elsewhere to recreate the workflow.

```bash
# Get the workflow input spec
comfy-pack run workflow.cpack.zip --help

# Run
comfy-pack run workflow.cpack.zip --src-image image.png --video video.mp4
```
</details>

<details> 
<summary> 4. (Optional) Deploy to the cloud </summary>

Deploy to [BentoCloud](https://www.bentoml.com/) with access to a variety of GPUs and blazing fast scaling.

Follow [the instructions here](https://docs.bentoml.com/en/latest/scale-with-bentocloud/manage-api-tokens.html) to get your BentoCloud access token. If you don’t have a BentoCloud account, you can [sign up for free](https://bentoml.com/).

![image](https://github.com/user-attachments/assets/1ffa31fc-1f50-4ea7-a47e-7dae3b874273)

</details>

## Roadmap

This project is under active development. Currently we are working on:

- Enhanced user experience
- Docker support
- Local `.cpack` file management with version control
- Enhanced service capabilities

## Community

comfy-pack is actively maintained by the BentoML team. Feel free to reach out 👉 [Join our Slack community!](https://l.bentoml.com/join-slack)

## Contributing

As an open-source project, we welcome contributions of all kinds, such as new features, bug fixes, and documentation. Here are some of the ways to contribute:

- Repost a bug by creating a [GitHub issue](https://github.com/bentoml/comfy-pack/issues).
- Submit a [pull request](https://github.com/bentoml/comfy-pack/pulls) or help review other developers’ pull requests.
