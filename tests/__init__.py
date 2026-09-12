"""SIH26059 trajectory module - test suite.

Run with:
    python -m unittest discover -s tests -v

All fixtures used across these tests are synthetic, generated in-memory
or in temporary directories (see tests/fixtures.py), and are NEVER
written into data/raw/, data/interim/, data/processed/, models/, or
outputs/. No test result here represents a real model metric or real
Antarctic trajectory - see test_no_fabrication.py for tests that
specifically guard against that distinction being lost.
"""
