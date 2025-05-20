import asyncio
import json
import re

from .const import MODEL_SOURCE_CACHE_FILE

# MODEL_NAME = r"[a-zA-Z0-9-._]+"
# COMMIT = r"[a-f0-9]+"

COMMIT_PATTERN = re.compile(r'href="/([a-zA-Z0-9-._/]+)/commit/([a-f0-9]+)"')

PATH_PATTERN = re.compile(
    r'data-target="CopyButton" data-props="{&quot;value&quot;:&quot;([^&]+)&quot;'
)


async def _lookup_huggingface_model(model_sha: str) -> dict:
    import aiohttp
    from duckduckgo_search import DDGS
    query = f"site:huggingface.co blob {model_sha}"

    try:
        with DDGS() as ddgs:
            search_results = ddgs.text(query, max_results=5)

            async with aiohttp.ClientSession(trust_env=True) as session:
                for result in search_results:
                    url = result['href']
                    if "blob" not in url:
                        continue

                    try:
                        async with session.get(url) as resp:
                            if resp.status != 200:
                                continue
                            text = await resp.text()
                            if commit_match := COMMIT_PATTERN.search(text):
                                repo, commit = commit_match.groups()
                                if path_match := PATH_PATTERN.search(text):
                                    path = path_match.group(1)
                                    info = {
                                        "download_url": path,
                                        "url": path,
                                        "repo": repo,
                                        "commit": commit,
                                        "path": path,
                                        "source": "huggingface",
                                    }
                                    return info
                    except aiohttp.ClientError:
                        continue
    except Exception:
        pass

    return {}


async def _loopup_civitai_model(model_sha: str) -> dict:
    import aiohttp

    async with aiohttp.ClientSession(trust_env=True) as session:
        async with session.post(
            "https://meilisearch-v1-9.civitai.com/multi-search",
            headers={
                "accept": "*/*",
                "accept-language": "en,zh;q=0.9,zh-CN;q=0.8",
                "cache-control": "no-cache",
                "content-type": "application/json",
                "origin": "https://civitai.com",
                "pragma": "no-cache",
                "priority": "u=1, i",
                "referer": "https://civitai.com/",
                "sec-ch-ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
                "sec-ch-ua-mobile": "?0",
                "sec-ch-ua-platform": '"macOS"',
                "sec-fetch-dest": "empty",
                "sec-fetch-mode": "cors",
                "sec-fetch-site": "same-site",
                "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
                "x-meilisearch-client": "Meilisearch instant-meilisearch (v0.13.5) ; Meilisearch JavaScript (v0.34.0)",
            },
            json={
                "queries": [
                    {
                        "q": model_sha,
                        "indexUid": "models_v9",
                        "facets": [
                            "category.name",
                            "checkpointType",
                            "fileFormats",
                            "lastVersionAtUnix",
                            "tags.name",
                            "type",
                            "user.username",
                            "version.baseModel",
                        ],
                        "attributesToHighlight": [],
                        "highlightPreTag": "__ais-highlight__",
                        "highlightPostTag": "__/ais-highlight__",
                        "limit": 51,
                        "offset": 0,
                        "filter": ["nsfwLevel=1"],
                    }
                ]
            },
        ) as resp:
            if resp.status != 200:
                return {}
            data = await resp.json()

            if len(data.get("results", [])) == 0:
                return {}

            if len(data["results"][0]["hits"]) == 0:
                return {}

            hit = data["results"][0]["hits"][0]
            repo_id = hit["id"]
            repo_name = hit["name"]
            versions = hit["versions"]

            for version in versions:
                if model_sha.upper() in version["hashes"]:
                    break
            else:
                return {}
            version_id = version["id"]
            version_name = version["name"]
            return {
                "download_url": f"https://civitai.com/api/download/models/{version_id}",
                "url": f"https://civitai.com/models/{repo_id}?modelVersionId={version_id}",
                "repo": repo_id,
                "commit": version_id,
                "source": "civitai",
                "repo_name": repo_name,
                "version_name": version_name,
            }


async def alookup_model_source(model_sha: str, cache_only=False) -> dict:
    if not model_sha:
        return {}
    try:
        model_source_cache = json.loads(MODEL_SOURCE_CACHE_FILE.read_text())
    except Exception:
        with open(MODEL_SOURCE_CACHE_FILE, "w") as f:
            f.write("{}")
        model_source_cache = {}
    if model_source_cache.get(model_sha):
        return model_source_cache[model_sha]
    if cache_only:
        return {}

    info = await _lookup_huggingface_model(model_sha)
    if not info:
        info = await _loopup_civitai_model(model_sha)

    # elemental read and write
    model_source_cache = json.loads(MODEL_SOURCE_CACHE_FILE.read_text())
    model_source_cache[model_sha] = info
    with open(MODEL_SOURCE_CACHE_FILE, "w") as f:
        json.dump(model_source_cache, f)

    return info


def lookup_model_source(model_sha: str, cache_only=False) -> dict:
    return asyncio.run(alookup_model_source(model_sha, cache_only=cache_only))
