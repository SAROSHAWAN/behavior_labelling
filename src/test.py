import pytest
from processData.textPipeline import iter_books, book_process, global_ent

def test_pipeline_with_real_data():
    """
    Test the pipeline using the actual files in data/book/test.
    """
    # 1. Clear global containers to start fresh for the test
    global_ent.clear()
    
    # 2. Get the first book from the test folder
    # iter_books yields (book_id, text)
    books = list(iter_books(mode="test"))
    
    # Ensure there is at least one file in data/book/test/
    assert len(books) > 0, "No .clean.txt files found in data/book/test"
    
    book_id, text = books[0]
    print(f"\nTesting with book: {book_id}")

    # 3. Process the book
    # This runs sliding_window, NER extraction, and coref logic
    doc_container = book_process(text)

    # 4. Verify Results
    # Check that we actually extracted data
    assert len(doc_container) > 0, "Pipeline failed to produce doc chunks"
    assert len(global_ent) > 0, "No PERSON entities were extracted from the test file"

    # 5. Schema Validation for the first entity
    sample_ent = global_ent[0]
    required_keys = [
        "type", "text", "global_start", "global_end", 
        "doc_id", "doc_token_pos", "sentence_id"
    ]
    
    for key in required_keys:
        assert key in sample_ent, f"Key '{key}' missing from extracted entity"

    # Verify doc_token_pos is a tuple for span access
    assert isinstance(sample_ent["doc_token_pos"], tuple)
    
    print(f"Test Passed: Extracted {len(global_ent)} entities from {book_id}")

if __name__ == "__main__":
    # Allows manual execution: python src/test.py
    test_pipeline_with_real_data()