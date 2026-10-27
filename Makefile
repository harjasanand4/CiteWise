.PHONY: install corpus index eval eval-answers ablation test lint web serve dev up clean

install:            ## Python package + dev tools, and the web UI's packages
	pip install -e ".[dev]"
	cd web && npm install

corpus:             ## Download both handbooks at their pinned commits
	citewise fetch

index: corpus       ## Parse, chunk, embed and index (incremental)
	citewise ingest

eval:               ## Retrieval metrics for every configuration + abstention sweep
	python -m evals.run --sweep

eval-answers:       ## Also generate answers (uses your LLM if one is configured in .env) and judge them
	python -m evals.run --answers --judge

ablation:           ## Same eval with static WordLlama embeddings, in a separate index (Python 3.11-3.13)
	pip install -e ".[lite]"
	CITEWISE_EMBEDDING_MODEL=wordllama CITEWISE_DATABASE_URL=sqlite:///data/wordllama.db citewise ingest
	CITEWISE_EMBEDDING_MODEL=wordllama CITEWISE_DATABASE_URL=sqlite:///data/wordllama.db python -m evals.run --configs dense,hybrid

test:
	pytest

lint:
	ruff check .
	mypy src

web:                ## Build the UI into web/dist (served by the API)
	cd web && npm run build

serve: web          ## API + UI on http://localhost:8000
	citewise serve

dev:                ## API with reload; run `cd web && npm run dev` in another terminal for the UI
	citewise serve --reload

up:                 ## Postgres + API in Docker
	docker compose up -d --build
	docker compose run --rm api sh -c "citewise fetch && citewise ingest"

clean:
	rm -rf data/*.db data/*.db-* evals/reports/latest.* web/dist
