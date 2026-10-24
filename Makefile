.PHONY: seed-play-history reset-play-history

seed-play-history:
	python3 scripts/seed_play_history.py

reset-play-history:
	rm -f "$${DATA_DIR:-$$HOME/.spot_ai_fy}/play_log.sqlite"
	$(MAKE) seed-play-history
