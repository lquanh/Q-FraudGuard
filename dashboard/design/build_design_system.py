"""Compile the dashboard's DTCG tokens with the project's UI/UX Pro Max dependency."""
from pathlib import Path
import subprocess


def main() -> None:
    folder = Path(__file__).resolve().parent
    compiler = folder / "node_modules/ui-ux-pro-max-cli/assets/skills/design-system/scripts/generate-tokens.cjs"
    if not compiler.is_file():
        raise SystemExit("Install the design tools with npm ci --prefix dashboard/design.")
    tokens = folder / "design-system/q-fraudguard/tokens.json"
    subprocess.run(
        ["node", str(compiler), "--config", str(tokens), "--output", str(tokens.with_suffix(".css"))],
        check=True,
    )


if __name__ == "__main__":
    main()
