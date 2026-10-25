.PHONY: eval eval-mock seed-library reset-library

DATA_DIR ?= $(HOME)/.spot_ai_fy

EVAL_ARGS ?= --thresholds

eval-mock:
	cd backend && python3 -m evals.run --mode mock --ci $(EVAL_ARGS)

eval:
	@./scripts/run-live-evals.sh

seed-library:
	@echo "SAMPLE_DATA — seeding fictional library catalog (not your Spotify account)"
	DATA_DIR="$(DATA_DIR)" python3 scripts/seed-sample-library.py

reset-library:
	@echo "SAMPLE_DATA — removing local sample library index and marker"
	DATA_DIR="$(DATA_DIR)" python3 scripts/reset-sample-library.py
