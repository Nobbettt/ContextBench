# SPDX-License-Identifier: Apache-2.0

"""Config loading and effective-variant expansion for run suites."""

from __future__ import annotations

import json
from pathlib import Path

from ..coding_agents.files import safe_path_component
from .helpers import deep_merge
from .types import (
    EffectiveVariantConfig,
    RunSuiteConfig,
    VariantConfig,
    VariantSetupConfig,
)


def merge_setup_config(base: VariantSetupConfig, override: VariantSetupConfig) -> VariantSetupConfig:
    return VariantSetupConfig(
        prompt_preamble=(
            override.prompt_preamble
            if override.prompt_preamble is not None
            else base.prompt_preamble
        ),
        setup_prompt=(
            override.setup_prompt
            if override.setup_prompt is not None
            else base.setup_prompt
        ),
        setup_prompt_timeout=(
            override.setup_prompt_timeout
            if override.setup_prompt_timeout is not None
            else base.setup_prompt_timeout
        ),
        copy_paths=[*base.copy_paths, *override.copy_paths],
        files_to_materialize=[*base.files_to_materialize, *override.files_to_materialize],
        claude_settings_overrides=deep_merge(
            base.claude_settings_overrides,
            override.claude_settings_overrides,
        ),
        claude_mcp_config=deep_merge(
            base.claude_mcp_config,
            override.claude_mcp_config,
        ),
    )


def build_run_suite_variant(
    run_suite: RunSuiteConfig,
    variant: VariantConfig,
) -> EffectiveVariantConfig:
    base = run_suite.base_run
    agent_args = list(variant.agent_args_replace) if variant.agent_args_replace is not None else [
        *base.agent_args,
        *variant.agent_args_add,
    ]
    env = dict(variant.env_replace) if variant.env_replace is not None else {**base.env, **variant.env_add}
    setup = merge_setup_config(base.setup, variant.setup)
    return EffectiveVariantConfig(
        name=variant.name,
        slug=safe_path_component(variant.name),
        description=variant.description,
        labels=list(variant.labels),
        notes=variant.notes,
        agent=run_suite.agent,
        task_data=base.task_data,
        task_csv=base.task_csv,
        subset_csv=base.subset_csv,
        bench=base.bench,
        instances=base.instances,
        limit=base.limit,
        timeout=variant.timeout or base.timeout,
        repo_cache=base.repo_cache,
        schema_path=base.schema_path,
        model=variant.model if variant.model is not None else base.model,
        reasoning_effort=(
            variant.reasoning_effort
            if variant.reasoning_effort is not None
            else base.reasoning_effort
        ),
        env=env,
        agent_args=agent_args,
        setup=setup,
    )


def load_run_suite_config(path: Path) -> RunSuiteConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return RunSuiteConfig.model_validate(payload)
