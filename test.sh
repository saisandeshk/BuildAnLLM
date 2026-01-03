#!/bin/bash

clear

# Playwright install (idempotent, checks for updates)
(cd frontend && npx playwright install)

echo "Running backend tests..."
uv run pytest

echo "Running frontend tests..."
(cd frontend && npm run test:run && npm run test:e2e)
