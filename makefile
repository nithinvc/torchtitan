format:
	find torchtitan/experiments/weather/ -type f -name "*.py" -print0 | xargs -0 uvx ruff format
install-torch-nightly:
	uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128 --force-reinstall