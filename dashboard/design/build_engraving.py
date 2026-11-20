"""Render the installed Taste Skill plate once for use in the dashboard masthead."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

from PIL import Image


PLATE = """<!doctype html><html lang="en"><meta charset="utf-8"><title>Engraving</title>
<style>html,body,main{margin:0;width:100%;height:100%;overflow:hidden;background:#183B56}</style>
<main></main><script type="module">
import fx from "./effects/fx.js";
await fx.mount(document.querySelector("main"), "line-engraving", {
  seed: 31,
  palette: {bg:"#183B56",surface:"#214765",ink:"#DCECF5",muted:"#4D91B8",accent:"#4D91B8",accentInk:"#183B56"},
  params: {source:1, pitch:6, angle:26, bend:1.2, minWidth:0.04, maxWidth:0.63,
           cross:0.15, guilloche:0.6, calm:0.18, calmX:0.14, calmY:0.52, drift:0}
});
</script></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    codex_skills = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")) / "skills"
    shared_skill = Path.home() / ".agents/skills/design-taste-frontend"
    profile_skill = codex_skills / "design-taste-frontend"
    default_skill = profile_skill if (profile_skill / "SKILL.md").is_file() else shared_skill
    parser.add_argument("--skill", type=Path, default=default_skill)
    skill = parser.parse_args().skill.resolve()
    if not (skill / "scripts/shoot.mjs").is_file():
        parser.error("Install design-taste-frontend or pass its directory with --skill.")
    effect_dir = skill / "assets/effects"
    output = Path(__file__).resolve().parents[1] / "assets/engraving.webp"
    with TemporaryDirectory(prefix="qfg-engraving-") as tmp:
        work = Path(tmp)
        plate = work / "plate.html"
        plate.write_text(PLATE, encoding="utf-8")
        (work / "effects/line-engraving").mkdir(parents=True)
        for file in ("fx.js", "line-engraving/effect.js"):
            shutil.copyfile(effect_dir / file, work / "effects" / file)
        subprocess.run(
            ["node", str(skill / "scripts/shoot.mjs"), "--url", str(plate), "--out", "plate.png",
             "--w", "1080", "--h", "200", "--wait", "1200", "--reduced", "--no-tiles"],
            cwd=work, check=True,
        )
        with Image.open(work / "plate.png") as shot:
            plate_rgb = shot.convert("RGB")
            if plate_rgb.size != (1080, 200) or max(hi - lo for lo, hi in plate_rgb.getextrema()) < 32:
                raise ValueError("The engraving did not render at the requested size and contrast.")
            output.parent.mkdir(exist_ok=True)
            plate_rgb.save(output, format="WEBP", quality=88, method=6)


if __name__ == "__main__":
    main()
