format:
	uvx ruff format torchtitan/experiments/weather/*

install-torch-nightly:
	uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128 --force-reinstall