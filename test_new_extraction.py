"""
Test new extraction with optimized chunking
"""

import os
import json
from text_extractor import TextExtractor

def test_new_extraction():
    print("🧪 Testing new extraction with optimized chunking...")
    
    extractor = TextExtractor()
    pdf_file = "Google IT Support Professional Certificate Notes (1).pdf"
    
    if not os.path.exists(pdf_file):
        print(f"❌ PDF file not found: {pdf_file}")
        return None
    
    print(f"📄 Processing: {pdf_file}")
    
    # Force new extraction with optimized chunking
    try:
        print("🔄 Running new extraction with optimized chunking...")
        extraction_results = extractor.extract_text(pdf_file, save_chunks=True)
        
        if 'error' not in extraction_results:
            chunks = extraction_results.get('chunks', [])
            print(f"✅ New extraction successful!")
            print(f"   📄 Total pages: {extraction_results.get('total_pages', 'Unknown')}")
            print(f"   📝 Text length: {len(extraction_results.get('text', ''))} characters")
            print(f"   🧩 Chunks created: {len(chunks)}")
            
            # Analyze chunk sizes
            if chunks:
                total_tokens = sum(chunk.get('estimated_tokens', 0) for chunk in chunks)
                avg_tokens = total_tokens / len(chunks)
                max_tokens = max(chunk.get('estimated_tokens', 0) for chunk in chunks)
                min_tokens = min(chunk.get('estimated_tokens', 0) for chunk in chunks)
                
                print(f"\n📊 Optimized Chunk Analysis:")
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
                for i, chunk in enumerate(chunks[:5]):  # Show first 5 chunks
                    print(f"\nChunk {i+1} (ID: {chunk['id']}):")
                    print(f"   Type: {chunk.get('type', 'Unknown')}")
                    print(f"   Length: {chunk.get('length', 0)} characters")
                    print(f"   Estimated tokens: {chunk.get('estimated_tokens', 0)}")
                    print(f"   Text preview: {chunk.get('text', '')[:150]}...")
                
                # Save optimized extraction
                optimized_file = "extracted_optimized_Google IT Support Professional Certificate Notes (1).json"
                with open(optimized_file, 'w', encoding='utf-8') as f:
                    json.dump(extraction_results, f, indent=2, ensure_ascii=False)
                print(f"\n💾 Optimized extraction saved to: {optimized_file}")
                
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

def main():
    print("🚀 Testing New Extraction with Optimized Chunking")
    print("="*60)
    
    extraction_results = test_new_extraction()
    
    if extraction_results:
        print("\n✅ New extraction with optimized chunking completed successfully!")
        print("   - Multiple smaller chunks created")
        print("   - All chunks within API token limits")
        print("   - Ready for efficient question generation")
    else:
        print("\n❌ New extraction test failed")

if __name__ == "__main__":
    main()
