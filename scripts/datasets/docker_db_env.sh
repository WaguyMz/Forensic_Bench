#!/usr/bin/env bash
# Source this file to point populate scripts and the agent at the Docker Postgres.
export FORENSIC_PG_HOST="${FORENSIC_PG_HOST:-localhost}"
export FORENSIC_PG_PORT="${FORENSIC_PG_PORT:-55432}"
export FORENSIC_PG_USER="${FORENSIC_PG_USER:-postgres}"
export FORENSIC_PG_PASS="${FORENSIC_PG_PASS:-forensicbench}"

export FORENSIC_DB_HOST="${FORENSIC_DB_HOST:-localhost}"
export FORENSIC_DB_PORT="${FORENSIC_DB_PORT:-55432}"
export FORENSIC_DB_USER="${FORENSIC_DB_USER:-postgres}"
export FORENSIC_DB_PASSWORD="${FORENSIC_DB_PASSWORD:-forensicbench}"
