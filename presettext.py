import hashlib
from pathlib import Path

from aiohttp import web
from aiohttp.web_request import Request
from pydantic import BaseModel
import yaml

from server import PromptServer


BASE_PATH = Path(__file__).parent.resolve()


class PresetText(BaseModel):
    name: str
    text: str


# Deja vu
class PresetTextLoader:
    LIST: list[PresetText]
    BY_NAME: dict[str, PresetText]
    CHOICES: list[str]

    def _get_presets_file(self) -> Path:
        presets_file = BASE_PATH / "texts.yaml"
        if not presets_file.exists():
            presets_file = BASE_PATH / "texts.default.yaml"
        return presets_file

    def load(self):
        presets_file = self._get_presets_file()

        with open(presets_file) as inp:
            data = yaml.load(inp, yaml.Loader)

        self._mtime = (presets_file, presets_file.stat().st_mtime)

        self.LIST = []
        for preset in data["presets"]:
            self.LIST.append(PresetText.model_validate(preset))
        self.BY_NAME = {p.name: p for p in self.LIST}
        self.CHOICES = [p.name for p in self.LIST]

    def refresh(self):
        presets_file = self._get_presets_file()
        if self._mtime != (presets_file, presets_file.stat().st_mtime):
            self.load()


LOADER = PresetTextLoader()
LOADER.load()


class PresetTextNode:
    @classmethod
    def INPUT_TYPES(cls):
        LOADER.refresh()
        return {
            "required": {
                "preset": (["None"] + LOADER.CHOICES,),
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                    },
                ),
            }
        }

    @classmethod
    def IS_CHANGED(cls, preset, text):
        # I feel like we can just return the text directly. We'll go with
        # this for now, I guess...
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return (h, len(text))

    TITLE = "Preset Text"

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("preset_text",)

    FUNCTION = "execute"

    CATEGORY = "private"

    def execute(self, preset, text):
        # It's all handled by the frontend, just pass the text on
        return (text,)


@PromptServer.instance.routes.get("/preset_text")
async def preset_text(request: Request):
    name = request.rel_url.query.get("name")
    if name:
        LOADER.refresh()
        if (preset := LOADER.BY_NAME.get(name)) is not None:
            return web.json_response(preset.text)

    # Fail, but don't error out
    return web.json_response("failed.to.fetch")


NODE_CLASS_MAPPINGS = {
    "PresetText": PresetTextNode,
}
