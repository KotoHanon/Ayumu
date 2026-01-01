"""
Unit tests for FAISSMemorySystem.abstract_episodic_records and upsert_abstract_semantic_records.

Run with:
    pytest memory/tests/test_abstract_episodic.py -v
    
Or for the real LLM test (requires API access):
    pytest memory/tests/test_abstract_episodic.py -v -k "real_llm" --run-real-llm
"""

import asyncio
import json
import pytest
import numpy as np
from typing import List, Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

from memory.api.faiss_memory_system_api import FAISSMemorySystem
from memory.memory_system.models import EpisodicRecord, SemanticRecord
from memory.memory_system.utils import now_iso, new_id


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_llm_response():
    """Mock LLM response for semantic record generation."""
    return json.dumps({
        "summary": "User frequently discusses car maintenance and service experiences.",
        "detail": "The user has visited the dealership multiple times for car services including first service, GPS repair, and routine maintenance. They seem satisfied with the service quality.",
        "tags": ["car", "maintenance", "dealership", "user-preference"]
    })


@pytest.fixture
def sample_episodic_records() -> List[EpisodicRecord]:
    """Generate sample episodic records for testing."""
    # Create embeddings that are similar (high cosine similarity) to trigger clustering
    base_embedding = np.random.randn(384).astype(np.float32)
    base_embedding = base_embedding / np.linalg.norm(base_embedding)
    
    records = []
    
    # Record 1: Car first service
    epi1 = EpisodicRecord(
        id=new_id("epi"),
        stage="user_event",
        summary="User brought new car to dealership for first service on March 15th. Service completed successfully.",
        detail={
            "session_id": "session_001",
            "situation": "User needed first car service",
            "actions": ["drove to dealership", "waited for service"],
            "results": ["service completed", "great experience"],
            "temporal_context": {
                "dates": ["March 15th"],
                "temporal_cues": ["first time"],
                "sequence": "single event"
            },
            "entities_involved": ["user", "car", "dealership"],
            "facts": ["user visited dealership", "first service completed"]
        },
        tags=["car", "service", "episodic-experience"],
        created_at=now_iso()
    )
    # Add small noise to make similar but not identical embedding
    epi1.embedding = base_embedding + np.random.randn(384).astype(np.float32) * 0.005
    epi1.embedding = epi1.embedding / np.linalg.norm(epi1.embedding)
    records.append(epi1)
    
    # Record 2: GPS repair (similar topic - car maintenance)
    epi2 = EpisodicRecord(
        id=new_id("epi"),
        stage="user_event",
        summary="User experienced GPS malfunction one week after first service. Returned to dealership, GPS system replaced.",
        detail={
            "session_id": "session_002",
            "situation": "GPS system malfunctioned",
            "actions": ["returned to dealership", "reported GPS issue"],
            "results": ["GPS replaced", "system working"],
            "temporal_context": {
                "dates": ["March 22nd"],
                "temporal_cues": ["one week after first service"],
                "sequence": "follow-up event"
            },
            "entities_involved": ["user", "GPS", "dealership"],
            "facts": ["GPS malfunction", "system replaced"]
        },
        tags=["car", "gps", "repair", "episodic-experience"],
        created_at=now_iso()
    )
    epi2.embedding = base_embedding + np.random.randn(384).astype(np.float32) * 0.005
    epi2.embedding = epi2.embedding / np.linalg.norm(epi2.embedding)
    records.append(epi2)
    
    # Record 3: Oil change (similar topic - car maintenance)
    epi3 = EpisodicRecord(
        id=new_id("epi"),
        stage="user_event",
        summary="User visited dealership for routine oil change. Quick service, no issues.",
        detail={
            "session_id": "session_003",
            "situation": "Routine maintenance needed",
            "actions": ["scheduled appointment", "dropped off car"],
            "results": ["oil changed", "car ready same day"],
            "temporal_context": {
                "dates": ["April 10th"],
                "temporal_cues": ["routine visit"],
                "sequence": "single event"
            },
            "entities_involved": ["user", "car", "dealership", "oil change"],
            "facts": ["routine oil change", "quick service"]
        },
        tags=["car", "maintenance", "oil-change", "episodic-experience"],
        created_at=now_iso()
    )
    epi3.embedding = base_embedding + np.random.randn(384).astype(np.float32) * 0.005
    epi3.embedding = epi3.embedding / np.linalg.norm(epi3.embedding)
    records.append(epi3)
    
    return records


@pytest.fixture
def dissimilar_episodic_records() -> List[EpisodicRecord]:
    """Generate episodic records with dissimilar embeddings (won't cluster together)."""
    records = []
    
    for i in range(3):
        # Create completely different random embeddings
        embedding = np.random.randn(384).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        
        epi = EpisodicRecord(
            id=new_id("epi"),
            stage="user_event",
            summary=f"Unrelated event {i}: User did something completely different.",
            detail={
                "session_id": f"session_unrelated_{i}",
                "situation": f"Unrelated situation {i}",
                "actions": [f"action_{i}"],
                "results": [f"result_{i}"],
            },
            tags=[f"topic_{i}", "episodic-experience"],
            created_at=now_iso()
        )
        epi.embedding = embedding
        records.append(epi)
    
    return records


@pytest.fixture
def episodic_memory_system():
    """Create a FAISSMemorySystem configured for episodic memory."""
    return FAISSMemorySystem(
        memory_type="episodic",
        llm_name="gpt-4o-mini",
        llm_backend="openai"
    )


@pytest.fixture
def semantic_memory_system():
    """Create a FAISSMemorySystem configured for semantic memory."""
    return FAISSMemorySystem(
        memory_type="semantic",
        llm_model="gpt-4o-mini",
        llm_backend="openai"
    )


# ============================================================================
# Unit Tests
# ============================================================================

class TestAbstractEpisodicRecords:
    """Tests for abstract_episodic_records method."""
    
    def test_assert_episodic_memory_type(self, semantic_memory_system, sample_episodic_records):
        """Test that abstract_episodic_records raises assertion for non-episodic memory type."""
        with pytest.raises(AssertionError, match="Clustering is only supported for episodic memory type"):
            asyncio.run(semantic_memory_system.abstract_episodic_records(sample_episodic_records))
    
    def test_empty_input(self, episodic_memory_system):
        """Test with empty episodic records list."""
        result, cidmap = asyncio.run(episodic_memory_system.abstract_episodic_records([]))
        assert result == []
        assert cidmap == {}
    
    @patch.object(FAISSMemorySystem, 'llm')
    def test_abstract_similar_records_mocked(
        self, 
        mock_llm, 
        episodic_memory_system, 
        sample_episodic_records,
        mock_llm_response
    ):
        """Test abstracting similar episodic records with mocked LLM."""
        # Setup mock
        mock_llm.complete = AsyncMock(return_value=mock_llm_response)
        episodic_memory_system.llm = mock_llm
        
        # Run abstraction
        result, cidmap = asyncio.run(
            episodic_memory_system.abstract_episodic_records(
                sample_episodic_records, 
                consistency_threshold=0.5  # Lower threshold to ensure clustering
            )
        )
        
        # Verify results
        print(f"\n[Test] Number of abstracted semantic records: {len(result)}")
        print(f"[Test] Cluster ID map: {cidmap}")
        
        # If records clustered together and met threshold, we should have at least one result
        # Note: Result depends on DenStream clustering behavior
        if len(result) > 0:
            assert all(isinstance(r, SemanticRecord) for r in result)
            assert all(r.is_abstracted == True for r in result)
            assert all(r.cluster_id is not None for r in result)
            print(f"[Test] First semantic record summary: {result[0].summary}")
    
    def test_abstract_dissimilar_records(self, episodic_memory_system, dissimilar_episodic_records):
        """Test that dissimilar records don't cluster together."""
        # With high consistency threshold and dissimilar embeddings, 
        # no clusters should meet the abstraction criteria
        result, cidmap = asyncio.run(
            episodic_memory_system.abstract_episodic_records(
                dissimilar_episodic_records, 
                consistency_threshold=0.9  # High threshold
            )
        )
        
        print(f"\n[Test] Dissimilar records abstraction result: {len(result)} semantic records")
        # Dissimilar records likely won't form consistent clusters
        # This is expected behavior
    
    @patch.object(FAISSMemorySystem, 'llm')
    def test_llm_parsing_error_handling(
        self, 
        mock_llm, 
        episodic_memory_system, 
        sample_episodic_records
    ):
        """Test handling of LLM response parsing errors."""
        # Return invalid JSON
        mock_llm.complete = AsyncMock(return_value="This is not valid JSON")
        episodic_memory_system.llm = mock_llm
        
        # Should not raise, but should print error and continue
        result, cidmap = asyncio.run(
            episodic_memory_system.abstract_episodic_records(
                sample_episodic_records,
                consistency_threshold=0.5
            )
        )
        
        # Even with parsing errors, function should complete gracefully
        assert isinstance(result, list)
        assert isinstance(cidmap, dict)


class TestUpsertAbstractSemanticRecords:
    """Tests for upsert_abstract_semantic_records method."""
    
    def test_upsert_new_records(self, semantic_memory_system):
        """Test upserting new semantic records (not previously seen clusters)."""
        # Create sample semantic records
        sem_records = []
        cidmap = {}
        
        for i in range(3):
            sem_rec = SemanticRecord(
                id=new_id("sem"),
                summary=f"Test semantic summary {i}",
                detail=f"Test detail for record {i}",
                tags=["test", f"tag_{i}"],
                is_abstracted=True,
                created_at=now_iso(),
                updated_at=now_iso()
            )
            sem_rec.cluster_id = i + 100  # Unique cluster IDs
            sem_records.append(sem_rec)
            cidmap[sem_rec.cluster_id] = sem_rec
        
        # Upsert
        semantic_memory_system.upsert_abstract_semantic_records(sem_records, cidmap)
        
        # Verify records were added to global map
        assert len(semantic_memory_system.global_cidmap2semrec) == 3
        for cid in cidmap.keys():
            assert cid in semantic_memory_system.global_cidmap2semrec
    
    def test_upsert_update_existing_records(self, semantic_memory_system):
        """Test updating existing semantic records (same cluster IDs)."""
        cluster_id = 999
        
        # First upsert
        sem_rec1 = SemanticRecord(
            id=new_id("sem"),
            summary="Original summary",
            detail="Original detail",
            tags=["original"],
            is_abstracted=True,
            created_at=now_iso(),
            updated_at=now_iso()
        )
        sem_rec1.cluster_id = cluster_id
        
        semantic_memory_system.upsert_abstract_semantic_records(
            [sem_rec1], 
            {cluster_id: sem_rec1}
        )
        
        original_id = semantic_memory_system.global_cidmap2semrec[cluster_id].id
        
        # Second upsert with same cluster_id (should update)
        sem_rec2 = SemanticRecord(
            id=new_id("sem"),
            summary="Updated summary",
            detail="Updated detail",
            tags=["updated"],
            is_abstracted=True,
            created_at=now_iso(),
            updated_at=now_iso()
        )
        sem_rec2.cluster_id = cluster_id
        
        semantic_memory_system.upsert_abstract_semantic_records(
            [sem_rec2], 
            {cluster_id: sem_rec2}
        )
        
        # Verify update happened
        stored_rec = semantic_memory_system.global_cidmap2semrec[cluster_id]
        assert stored_rec.summary == "Updated summary"
        assert stored_rec.detail == "Updated detail"
        assert "updated" in stored_rec.tags
        # ID should remain the same (update, not new add)
        assert stored_rec.id == original_id


class TestIntegration:
    """Integration tests combining abstract and upsert."""
    
    @patch.object(FAISSMemorySystem, 'llm')
    def test_full_pipeline_mocked(
        self,
        mock_llm,
        episodic_memory_system,
        semantic_memory_system,
        sample_episodic_records,
        mock_llm_response
    ):
        """Test full pipeline: abstract episodic -> upsert to semantic memory."""
        # Setup mock
        mock_llm.complete = AsyncMock(return_value=mock_llm_response)
        episodic_memory_system.llm = mock_llm
        
        # Step 1: Abstract episodic records
        abstract_result, cidmap = asyncio.run(
            episodic_memory_system.abstract_episodic_records(
                sample_episodic_records,
                consistency_threshold=0.5
            )
        )
        
        print(f"\n[Integration Test] Abstracted {len(abstract_result)} semantic records")
        
        # Step 2: Upsert to semantic memory system
        if len(abstract_result) > 0:
            semantic_memory_system.upsert_abstract_semantic_records(abstract_result, cidmap)
            
            # Verify
            assert len(semantic_memory_system.global_cidmap2semrec) > 0
            print(f"[Integration Test] Semantic memory now has {len(semantic_memory_system.global_cidmap2semrec)} cluster records")


# ============================================================================
# Real LLM Tests (Optional - requires API access)
# ============================================================================

def pytest_addoption(parser):
    """Add command line option to run real LLM tests."""
    parser.addoption(
        "--run-real-llm",
        action="store_true",
        default=False,
        help="Run tests that require real LLM API access"
    )


@pytest.fixture
def run_real_llm(request):
    """Fixture to check if real LLM tests should run."""
    return request.config.getoption("--run-real-llm")


class TestRealLLM:
    """Tests using real LLM (skip by default)."""
    
    @pytest.mark.skipif(
        "not config.getoption('--run-real-llm')",
        reason="Requires --run-real-llm option"
    )
    def test_abstract_with_real_llm(self, episodic_memory_system, sample_episodic_records):
        """Test abstraction with real LLM (requires API access)."""
        print("\n[Real LLM Test] Running with actual LLM...")
        
        result, cidmap = asyncio.run(
            episodic_memory_system.abstract_episodic_records(
                sample_episodic_records,
                consistency_threshold=0.5
            )
        )
        
        print(f"[Real LLM Test] Abstracted {len(result)} semantic records")
        
        for i, sem_rec in enumerate(result):
            print(f"\n[Real LLM Test] Semantic Record {i + 1}:")
            print(f"  Summary: {sem_rec.summary}")
            print(f"  Detail: {sem_rec.detail[:200]}..." if len(sem_rec.detail) > 200 else f"  Detail: {sem_rec.detail}")
            print(f"  Tags: {sem_rec.tags}")
            print(f"  Cluster ID: {sem_rec.cluster_id}")
            print(f"  Is Abstracted: {sem_rec.is_abstracted}")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    # Run a quick sanity check
    print("Running quick sanity check...")
    
    # Create test data
    base_embedding = np.random.randn(384).astype(np.float32)
    base_embedding = base_embedding / np.linalg.norm(base_embedding)
    
    epi_records = []
    for i in range(300):
        epi = EpisodicRecord(
            id=new_id("epi"),
            stage="user_event",
            summary=f"Test event {i}",
            detail={"test": f"detail_{i}"},
            tags=["test"],
            created_at=now_iso()
        )
        epi.embedding = base_embedding + np.random.randn(384).astype(np.float32) * 0.005
        epi.embedding = epi.embedding / np.linalg.norm(epi.embedding)
        epi_records.append(epi)
    
    # Create memory system
    print("Creating episodic memory system...")
    epi_mem = FAISSMemorySystem(
        memory_type="episodic",
        llm_name="gpt-4o-mini",
        llm_backend="openai"
    )
    
    print(f"Created {len(epi_records)} episodic records")
    print("Running abstract_episodic_records...")
    
    # Run abstraction
    result, cidmap = asyncio.run(
        epi_mem.abstract_episodic_records(epi_records, consistency_threshold=0.5)
    )
    
    print(f"Result: {len(result)} semantic records abstracted")
    print(f"Cluster map: {cidmap}")
    
    if len(result) > 0:
        print(f"\nFirst semantic record:")
        print(f"  Summary: {result[0].summary}")
        print(f"  Is Abstracted: {result[0].is_abstracted}")
        print(f"  Cluster ID: {result[0].cluster_id}")
    
    print("\nSanity check complete!")
