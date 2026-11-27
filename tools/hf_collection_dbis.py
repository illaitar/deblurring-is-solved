"""The Hugging Face collection "dbis": every hosted model repository of dbis.MODELS (run where you are logged in).

    python tools/hf_collection_dbis.py [--dry]

Creates the collection under dbis.HF_OWNER if it is not there yet and adds every model whose repository exists, in the
order of dbis.MODELS, each with a one-line note; prints the collection's address (dbis.COLLECTION).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from dbis import HF_OWNER, MODELS, SOURCES, repo  # noqa: E402

TITLE = "dbis"
ABOUT = "Deblurring networks in plain PyTorch: dbis = the authors' outputs, dbisf = faster and lighter"


def note(name):
    src = SOURCES[MODELS[name]["source"]]
    kind = "fast and light" if name.startswith("dbisf-") else "the authors' outputs"
    return f"{src['method']} ({src['venue']}), {kind}"


if __name__ == "__main__":
    dry = "--dry" in sys.argv
    names = [n for n, m in MODELS.items() if m["sha256"]]
    for n in names:
        print(f"{repo(n)}: {note(n)}")
    if dry:
        sys.exit(0)
    import huggingface_hub
    from huggingface_hub import HfApi

    import hf_push
    huggingface_hub.set_client_factory(hf_push._client_factory)
    api = HfApi()
    col = hf_push._retry("create_collection", lambda: api.create_collection(
        TITLE, namespace=HF_OWNER, description=ABOUT, exists_ok=True))
    have = {it.item_id for it in col.items}
    for n in names:
        if repo(n) in have:
            continue
        if not hf_push._retry(f"repo {repo(n)}", lambda n=n: api.repo_exists(repo(n))):
            print(f"  {repo(n)}: no repository yet, left out")
            continue
        hf_push._retry(f"add {repo(n)}", lambda n=n: api.add_collection_item(
            col.slug, repo(n), "model", note=note(n), exists_ok=True))
        print(f"  added {repo(n)}")
    print(f"collection {col.slug}: https://huggingface.co/collections/{col.slug}")
