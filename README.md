# AFTER

**AI-assisted FAST Transient End-to-end Reduction**

AFTER is an AI-assisted post-search FAST FRB burst processing workflow. It is designed for observations where the source, date, beam, DM, and burst TOA list are already known, then guides the workflow through TOA-guided cutting, calibration, burst-label review, energy/polarization/DM/RM analysis, and results-table export.

This initial repository upload publishes the Codex skill first. The processing scripts and model/data assets are not included in this first push.

## Codex Skill

The skill is available at:

```text
skills/fast-frb-observation-processing/
```

One-line install request for a Codex agent:

```text
Please install the Codex skill from this repository: copy skills/fast-frb-observation-processing into the Codex skills directory, set DATA_PROCESSING_ROOT to the local AFTER/data_processing script root when available, and run the post-install validation described by the skill.
```

Chinese prompt:

```text
请帮我安装本仓库的 Codex skill：复制 skills/fast-frb-observation-processing 到 Codex skills 目录；如果本机已有 AFTER/data_processing 脚本仓库，请把 DATA_PROCESSING_ROOT 设置为该脚本根目录，并完成安装后自检。
```

## Current Status

- Published now: Codex skill protocol and agent metadata.
- Not published yet: processing scripts, model weights, calibration assets, example data, and observation catalogs.

