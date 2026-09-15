"""Configuration loading, including the shipped config files."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from exist2026.config import (
    CONFIG_DIR,
    ROOT_ENV_VAR,
    ConfigError,
    load_fewshot_config,
    load_fusion_config,
    resolve_project_root,
)
from exist2026.taxonomy import Modality


@pytest.fixture(autouse=True)
def fake_root(tmp_path, monkeypatch):
    monkeypatch.setenv(ROOT_ENV_VAR, str(tmp_path))
    return tmp_path


class TestProjectRoot:
    def test_the_environment_variable_wins(self, fake_root):
        assert resolve_project_root("/somewhere/else") == fake_root

    def test_a_missing_root_is_an_error(self, monkeypatch):
        monkeypatch.delenv(ROOT_ENV_VAR, raising=False)
        with pytest.raises(ConfigError, match=ROOT_ENV_VAR):
            resolve_project_root(None)


class TestShippedConfigs:
    @pytest.mark.parametrize("name", ["fewshot_memes", "fewshot_videos"])
    def test_few_shot_configs_load(self, name):
        config = load_fewshot_config(name)
        assert config.modality in Modality
        assert set(config.pools) == set(config.subtask_keys)
        assert set(config.generation.max_new_tokens) == set(config.subtask_keys)

    @pytest.mark.parametrize("name", ["fusion_memes", "fusion_videos"])
    def test_fusion_configs_load(self, name):
        config = load_fusion_config(name)
        assert config.physio_kind in ("flat", "matrix")
        assert config.text_model

    def test_every_config_file_is_loadable(self):
        for path in sorted(CONFIG_DIR.glob("*.yaml")):
            loader = load_fewshot_config if path.stem.startswith("fewshot") else load_fusion_config
            loader(path)

    def test_meme_and_video_pools_differ_as_designed(self):
        memes = load_fewshot_config("fewshot_memes")
        videos = load_fewshot_config("fewshot_videos")
        assert not memes.pools["t21"].bilingual
        assert videos.pools["t31"].bilingual
        assert videos.pools["t33"].cooccurrence_extra == 2

    def test_official_thresholds_follow_the_modality(self):
        assert load_fewshot_config("fewshot_memes").thresholds.identification == 3
        assert load_fewshot_config("fewshot_videos").thresholds.identification == 1


class TestStages:
    def test_sanity_runs_score_a_training_subset(self):
        config = load_fewshot_config("fewshot_memes", stage="sanity")
        assert config.run_mode == "train_eval"
        assert config.max_eval_instances is not None
        assert config.checkpoint_suffix.startswith("sanity")

    def test_submission_runs_use_the_whole_test_split(self):
        config = load_fewshot_config("fewshot_memes", stage="submission")
        assert config.run_mode == "test"
        assert config.max_eval_instances is None
        assert config.checkpoint_suffix == "test"

    def test_an_unknown_stage_is_rejected(self):
        with pytest.raises(ConfigError, match="stage"):
            load_fewshot_config("fewshot_memes", stage="production")

    def test_an_unknown_backend_is_rejected(self):
        with pytest.raises(ConfigError, match="unknown backend"):
            load_fewshot_config("fewshot_memes", backend="llama9")

    def test_backends_can_be_swapped(self):
        config = load_fewshot_config("fewshot_memes", backend="qwen35")
        assert config.backend.needs_qwen_vl_utils
        assert "Qwen" in config.backend.model_id


class TestPaths:
    def test_dataset_paths_resolve_under_the_root(self, fake_root):
        config = load_fewshot_config("fewshot_memes")
        assert str(config.data.train_json_candidates[0]).startswith(str(fake_root))

    def test_a_missing_dataset_names_every_candidate(self):
        config = load_fewshot_config("fewshot_memes")
        with pytest.raises(FileNotFoundError, match="training JSON"):
            config.data.train_json()

    def test_optional_paths_may_be_absent(self):
        assert load_fusion_config("fusion_memes").data.vlm_json() is None

    def test_run_directories_are_created_on_demand(self, fake_root):
        dirs = load_fewshot_config("fewshot_memes").dirs.create()
        assert dirs.pools.is_dir()
        assert dirs.submission.is_dir()


def test_a_missing_config_file_is_an_error():
    with pytest.raises(FileNotFoundError):
        load_fewshot_config(Path("no_such_config.yaml"))


def test_config_dir_is_inside_the_repository():
    assert CONFIG_DIR.is_dir()
    assert os.path.basename(CONFIG_DIR) == "configs"
