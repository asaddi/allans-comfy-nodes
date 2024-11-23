from pathlib import Path

from huggingface_hub import snapshot_download
from pydantic import BaseModel
import yaml

import folder_paths


BASE_NAME = Path(__file__).parent.resolve()


class ModelDefinition(BaseModel):
    name: str
    repo_id: str
    revision: str | None = None
    use_hf_cache: bool


class ModelManager:
    LIST: list[ModelDefinition] = []
    BY_NAME: dict[str, ModelDefinition] = {}
    CHOICES: list[str] = []

    _MODELS_SUBDIR = "LLM"
    _ALLOWED_FILES = ["*.json", "*.safetensors"]
    _CONFIG = "models.yaml"
    _CONFIG_DEFAULT = "models.yaml.default"

    def __init__(
        self,
        models_subdir: str | None = None,
        allowed_files: list[str] | None = None,
        config_file: str | None = None,
        default_config_file: str | None = None,
    ):
        if models_subdir is not None:
            self._MODELS_SUBDIR = models_subdir
        if allowed_files is not None:
            self._ALLOWED_FILES = allowed_files
        if config_file is not None:
            self._CONFIG = config_file
        if default_config_file is not None:
            self._CONFIG_DEFAULT = default_config_file

    def _get_models_file(self) -> Path:
        models_file = Path(BASE_NAME) / self._CONFIG
        if not models_file.exists():
            models_file = Path(BASE_NAME) / self._CONFIG_DEFAULT
        return models_file

    def load(self):
        models_file = self._get_models_file()

        with open(models_file) as inp:
            d = yaml.load(inp, yaml.Loader)
        self._mtime = (models_file, models_file.stat().st_mtime)

        self.LIST = []
        for value in d["models"]:
            self.LIST.append(ModelDefinition.model_validate(value))
        if not self.LIST:
            raise RuntimeError("Need at least one model defined")
        self.BY_NAME = {d.name: d for d in self.LIST}
        self.CHOICES = [d.name for d in self.LIST]

    def refresh(self):
        models_file = self._get_models_file()
        if self._mtime != (models_file, models_file.stat().st_mtime):
            self.load()

    def download(self, name: str) -> Path:
        model_def = self.BY_NAME[name]

        if (as_dir := Path(model_def.repo_id)).is_dir():
            # Local path, nothing to do
            return as_dir

        if model_def.use_hf_cache:
            # Easy peasy
            return Path(
                snapshot_download(
                    model_def.repo_id,
                    revision=model_def.revision,
                    allow_patterns=self._ALLOWED_FILES,
                )
            )
        else:
            dir_name = "--".join(model_def.repo_id.split("/"))
            model_path = Path(folder_paths.models_dir) / self._MODELS_SUBDIR / dir_name
            model_path.mkdir(parents=True, exist_ok=True)
            return Path(
                snapshot_download(
                    model_def.repo_id,
                    revision=model_def.revision,
                    allow_patterns=self._ALLOWED_FILES,
                    local_dir=model_path,
                )
            )
