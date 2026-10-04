import pytest

from core import orchestrator
from core.schemas import get_db_path

pytestmark = pytest.mark.integration


@pytest.mark.skipif(not get_db_path().exists(), reason="local knowledge base not built")
def test_run_query_end_to_end_on_real_data():
    out = orchestrator.run_query("NVDA", "revenue growth margin")

    assert out["agents_activated"] == ["FinancialAgent"]
    result = out["agent_results"]["FinancialAgent"]
    assert result.status == "success"
    assert result.findings
    assert isinstance(out["verification"], list)
