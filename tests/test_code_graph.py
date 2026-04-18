"""
tests/test_code_graph.py — Unit and integration tests for Code Review Graph.
"""

import pytest
import tempfile
from pathlib import Path
from core.code_graph import CodeGraphManager

@pytest.fixture
def temp_workspace():
    """Create a temporary workspace with some Python files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        
        # File 1: utils.py
        utils_py = root / "utils.py"
        utils_py.write_text("""
def helper_func(x):
    \"\"\"Docstring for helper.\"\"\"
    return x * 2

class Utility:
    def __init__(self):
        pass
    def process(self):
        return helper_func(10)
""", encoding="utf-8")

        # File 2: main.py
        main_py = root / "main.py"
        main_py.write_text("""
from utils import helper_func, Utility

def main():
    u = Utility()
    result = u.process()
    print(helper_func(result))

if __name__ == "__main__":
    main()
""", encoding="utf-8")

        yield str(root)

@pytest.mark.asyncio
async def test_indexing_extracts_symbols(temp_workspace):
    manager = CodeGraphManager(temp_workspace)
    await manager.index_codebase(force=True)
    
    # Check nodes
    assert manager.graph.has_node("utils")
    assert manager.graph.has_node("main")
    assert manager.graph.has_node("utils:Utility")
    assert manager.graph.has_node("utils.helper_func")
    assert manager.graph.has_node("main.main")
    
    # Check data
    data = manager.graph.nodes["utils.helper_func"]
    assert data["type"] == "function"
    assert data["name"] == "helper_func"
    assert "Docstring for helper" in data["docstring"]

@pytest.mark.asyncio
async def test_indexing_extracts_relationships(temp_workspace):
    manager = CodeGraphManager(temp_workspace)
    await manager.index_codebase(force=True)
    
    # check inherits
    # check calls
    assert manager.graph.has_edge("main.main", "helper_func")
    assert manager.graph.get_edge_data("main.main", "helper_func")["type"] == "calls"
    
    assert manager.graph.has_edge("utils:Utility.process", "helper_func")
    assert manager.graph.get_edge_data("utils:Utility.process", "helper_func")["type"] == "calls"

    assert manager.graph.has_edge("main", "utils.helper_func")
    assert manager.graph.get_edge_data("main", "utils.helper_func")["type"] == "imports"

@pytest.mark.asyncio
async def test_impact_analysis(temp_workspace):
    manager = CodeGraphManager(temp_workspace)
    await manager.index_codebase(force=True)
    
    # If helper_func changes, what is impacted?
    # In my simplified AST call edge logic, we call it 'helper_func' (Name node)
    # but we also have the full id 'utils.helper_func'.
    # The edges currently use the नेम used in the call (e.g. 'helper_func').
    
    impact = manager.get_impact_analysis("helper_func")
    assert impact["impacted_symbols_count"] >= 2
    assert "main.py" in impact["impacted_files"]
    assert "utils.py" in impact["impacted_files"]

@pytest.mark.asyncio
async def test_search_symbols(temp_workspace):
    manager = CodeGraphManager(temp_workspace)
    await manager.index_codebase(force=True)
    
    results = manager.find_symbol("Utility")
    assert len(results) >= 1
    assert any(r["id"] == "utils:Utility" for r in results)

@pytest.mark.asyncio
async def test_incremental_indexing(temp_workspace):
    manager = CodeGraphManager(temp_workspace)
    await manager.index_codebase(force=True)
    initial_node_count = manager.graph.number_of_nodes()
    
    # Add a new file
    extra_py = Path(temp_workspace) / "extra.py"
    extra_py.write_text("def extra(): pass", encoding="utf-8")
    
    await manager.index_codebase(force=False)
    assert manager.graph.number_of_nodes() == initial_node_count + 2 # module + function
    assert manager.graph.has_node("extra.extra")
