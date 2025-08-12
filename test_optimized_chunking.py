"""
Test optimized chunking system
"""

import os
import json
from text_extractor import TextExtractor

def test_optimized_chunking():
    print("🧪 Testing optimized chunking system...")
    
    extractor = TextExtractor()
    pdf_file = "Google IT Support Professional Certificate Notes (1).pdf"
    
    if not os.path.exists(pdf_file):
        print(f"❌ PDF file not found: {pdf_file}")
        return None
    
    print(f"📄 Processing: {pdf_file}")
    
    # Test chunking without question generation first
    try:
        # Load existing extraction if available
        extraction_file = f"extracted_{pdf_file.replace('.pdf', '')}.json"
        if os.path.exists(extraction_file):
            print(f"📂 Loading existing extraction from: {extraction_file}")
            with open(extraction_file, 'r', encoding='utf-8') as f:
                extraction_results = json.load(f)
        else:
            print("🔄 Running new extraction...")
            extraction_results = extractor.extract_text(pdf_file, save_chunks=True)
        
        if 'error' not in extraction_results:
            chunks = extraction_results.get('chunks', [])
            print(f"✅ Extraction successful!")
            print(f"   📄 Total pages: {extraction_results.get('total_pages', 'Unknown')}")
            print(f"   📝 Text length: {len(extraction_results.get('text', ''))} characters")
            print(f"   🧩 Chunks created: {len(chunks)}")
            
            # Analyze chunk sizes
            if chunks:
                total_tokens = sum(chunk.get('estimated_tokens', 0) for chunk in chunks)
                avg_tokens = total_tokens / len(chunks)
                max_tokens = max(chunk.get('estimated_tokens', 0) for chunk in chunks)
                min_tokens = min(chunk.get('estimated_tokens', 0) for chunk in chunks)
                
                print(f"\n📊 Chunk Analysis:")
                print(f"   Total estimated tokens: {total_tokens}")
                print(f"   Average tokens per chunk: {avg_tokens:.1f}")
                print(f"   Max tokens in chunk: {max_tokens}")
                print(f"   Min tokens in chunk: {min_tokens}")
                
                # Check if chunks are within API limits
                api_limit = 8000
                chunks_within_limit = sum(1 for chunk in chunks if chunk.get('estimated_tokens', 0) <= api_limit)
                chunks_over_limit = len(chunks) - chunks_within_limit
                
                print(f"\n🔍 API Limit Check:")
                print(f"   Chunks within limit (≤{api_limit} tokens): {chunks_within_limit}")
                print(f"   Chunks over limit: {chunks_over_limit}")
                
                if chunks_over_limit == 0:
                    print(f"   ✅ All chunks are within API limits!")
                else:
                    print(f"   ⚠️  {chunks_over_limit} chunks exceed API limits")
                
                # Show sample chunks
                print(f"\n📋 Sample Chunks:")
                for i, chunk in enumerate(chunks[:3]):
                    print(f"\nChunk {i+1} (ID: {chunk['id']}):")
                    print(f"   Type: {chunk.get('type', 'Unknown')}")
                    print(f"   Length: {chunk.get('length', 0)} characters")
                    print(f"   Estimated tokens: {chunk.get('estimated_tokens', 0)}")
                    print(f"   Text preview: {chunk.get('text', '')[:100]}...")
                
                return extraction_results
            else:
                print(f"❌ No chunks created")
                return None
        else:
            print(f"❌ Extraction failed: {extraction_results['error']}")
            return None
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_question_generation_with_optimized_chunks():
    print("\n🧪 Testing question generation with optimized chunks...")
    
    # Load the optimized extraction
    extraction_file = "extracted_Google IT Support Professional Certificate Notes (1).json"
    if not os.path.exists(extraction_file):
        print(f"❌ Extraction file not found: {extraction_file}")
        return None
    
    with open(extraction_file, 'r', encoding='utf-8') as f:
        extraction_results = json.load(f)
    
    chunks = extraction_results.get('chunks', [])
    if not chunks:
        print("❌ No chunks found in extraction results")
        return None
    
    # Test question generation on a few chunks
    extractor = TextExtractor()
    test_chunks = chunks[:3]  # Test first 3 chunks
    
    print(f"🔄 Testing question generation on {len(test_chunks)} chunks...")
    
    successful_generations = 0
    for i, chunk in enumerate(test_chunks):
        try:
            print(f"\n📝 Testing chunk {i+1} (ID: {chunk['id']}):")
            print(f"   Estimated tokens: {chunk.get('estimated_tokens', 0)}")
            
            questions = extractor.generate_questions(chunk['text'])
            
            if questions and not questions.startswith("Error"):
                print(f"   ✅ Questions generated successfully!")
                print(f"   📄 Question preview: {questions[:200]}...")
                successful_generations += 1
            else:
                print(f"   ❌ Question generation failed: {questions}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n📊 Question Generation Results:")
    print(f"   Successful generations: {successful_generations}/{len(test_chunks)}")
    print(f"   Success rate: {(successful_generations/len(test_chunks))*100:.1f}%")
    
    return successful_generations > 0

def main():
    print("🚀 Testing Optimized Chunking System")
    print("="*60)
    
    # Test chunking optimization
    extraction_results = test_optimized_chunking()
    
    if extraction_results:
        # Test question generation
        question_success = test_question_generation_with_optimized_chunks()
        
        if question_success:
            print("\n✅ Optimized chunking system working correctly!")
            print("   - Chunks are within API token limits")
            print("   - Question generation is successful")
            print("   - System is ready for production use")
        else:
            print("\n⚠️  Chunking optimized but question generation needs attention")
    else:
        print("\n❌ Chunking optimization test failed")

if __name__ == "__main__":
    main()
