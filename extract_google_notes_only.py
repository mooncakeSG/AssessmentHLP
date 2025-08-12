"""
Extract Google Notes Only
Extracts text and creates chunks from Google IT Support PDF without generating questions
"""

import os
import json
import logging
from text_extractor import TextExtractor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_google_notes_only():
    """Extract Google notes without question generation"""
    print("🚀 Extracting Google IT Support Notes")
    print("="*50)
    
    pdf_file = "Google IT Support Professional Certificate Notes (1).pdf"
    
    if not os.path.exists(pdf_file):
        print(f"❌ PDF file not found: {pdf_file}")
        return None
    
    print(f"📄 Processing: {pdf_file}")
    
    try:
        # Create extractor
        extractor = TextExtractor()
        
        # Extract text and create chunks (without question generation)
        print("🔄 Extracting text and creating chunks...")
        
        # Temporarily modify the extract_text method to skip question generation
        original_extract_text = extractor.extract_text
        
        def extract_text_no_questions(pdf_path: str, save_chunks: bool = True):
            """Extract text without generating questions"""
            logger.info(f"Starting text extraction from: {pdf_path}")
            pdf_type = extractor.detect_pdf_type(pdf_path)
            logger.info(f"Detected PDF type: {pdf_type}")

            if pdf_type == 'searchable':
                extraction_result = extractor.extract_text_searchable(pdf_path)
            elif pdf_type == 'scanned':
                extraction_result = extractor.extract_text_scanned_parallel(pdf_path)
            else:
                extraction_result = extractor.extract_text_searchable(pdf_path)
                if not extraction_result.get('text'):
                    extraction_result = extractor.extract_text_scanned_parallel(pdf_path)

            if 'error' in extraction_result:
                logger.error(f"Extraction failed: {extraction_result['error']}")
                return extraction_result

            chunks = extractor.segment_content(extraction_result['text'])
            logger.info(f"Created {len(chunks)} chunks from text")
            
            # Log chunk statistics
            total_tokens = sum(chunk.get('estimated_tokens', 0) for chunk in chunks)
            avg_tokens = total_tokens / len(chunks) if chunks else 0
            logger.info(f"Total estimated tokens: {total_tokens}, Average per chunk: {avg_tokens:.1f}")
            
            chunks = extractor.embed_chunks(chunks)
            extraction_result['chunks'] = chunks

            if save_chunks:
                extractor._save_extraction_results(pdf_path, extraction_result)

            logger.info(f"Extraction completed. {len(chunks)} chunks created.")
            return extraction_result
        
        # Use the modified method
        extractor.extract_text = extract_text_no_questions
        
        # Extract the PDF
        results = extractor.extract_text(pdf_file, save_chunks=True)
        
        if 'error' not in results:
            chunks = results.get('chunks', [])
            print(f"✅ Extraction successful!")
            print(f"📄 Total pages: {results.get('total_pages', 'Unknown')}")
            print(f"📝 Text length: {len(results.get('text', ''))} characters")
            print(f"🔧 Method: {results.get('method', 'Unknown')}")
            print(f"📊 Quality: {results.get('extraction_quality', 'Unknown')}")
            print(f"🧩 Chunks created: {len(chunks)}")
            
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
            
            return results
        else:
            print(f"❌ Extraction failed: {results['error']}")
            return None
            
    except Exception as e:
        print(f"❌ Error during extraction: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function"""
    results = extract_google_notes_only()
    
    if results:
        print(f"\n🎉 Google notes extraction completed successfully!")
        print(f"📁 Output file: extracted_Google IT Support Professional Certificate Notes (1).json")
        print(f"🔧 Ready for cross-reference question generation!")
    else:
        print(f"\n❌ Extraction failed")

if __name__ == "__main__":
    main()
