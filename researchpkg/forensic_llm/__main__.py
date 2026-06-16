"""Allow ``python -m researchpkg.forensic_llm``."""
import sys

from researchpkg.forensic_llm.run import main

if __name__ == "__main__":
    sys.exit(main())
