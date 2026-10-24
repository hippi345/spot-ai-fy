.PHONY: eval eval-mock

eval-mock:
	cd backend && python3 -m evals.run --mode mock --ci --regression-report --thresholds

eval:
	@./scripts/run-live-evals.sh
