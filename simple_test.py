"""
Simple test to check basic functionality
"""

import os
import json
from text_extractor import TextExtractor
from content_analyzer import ContentAnalyzer

def test_basic_extraction():
    print("🧪 Testing basic text extraction...")
    
    extractor = TextExtractor()
    pdf_file = "Google IT Support Professional Certificate Notes (1).pdf"
    
    if os.path.exists(pdf_file):
        print(f"📄 Processing: {pdf_file}")
        
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
            print(f"✅ Extraction successful!")
            print(f"   📄 Total pages: {extraction_results.get('total_pages', 'Unknown')}")
            print(f"   📝 Text length: {len(extraction_results.get('text', ''))} characters")
            print(f"   🧩 Chunks: {len(extraction_results.get('chunks', []))}")
            
            return extraction_results
        else:
            print(f"❌ Extraction failed: {extraction_results['error']}")
            return None
    else:
        print(f"⚠️  PDF file not found: {pdf_file}")
        return None

def test_basic_analysis(extraction_results):
    print("\n🧪 Testing basic content analysis...")
    
    if not extraction_results:
        print("❌ No extraction results to analyze")
        return None
    
    analyzer = ContentAnalyzer()
    analysis = analyzer.analyze_content(extraction_results)
    
    if 'error' not in analysis:
        print(f"✅ Analysis successful!")
        print(f"   📊 Concepts found: {len(analysis.get('concepts', []))}")
        print(f"   📚 Topics identified: {len(analysis.get('topics', []))}")
        print(f"   📝 Facts extracted: {len(analysis.get('facts', []))}")
        
        # Save analysis results
        output_file = "analysis_test.json"
        analyzer.save_analysis_results(analysis, output_file)
        print(f"   💾 Saved to: {output_file}")
        
        return analysis
    else:
        print(f"❌ Analysis failed: {analysis['error']}")
        return None

def main():
    print("🚀 Starting simple system test")
    print("="*50)
    
    # Test extraction
    extraction_results = test_basic_extraction()
    
    # Test analysis
    if extraction_results:
        analysis_results = test_basic_analysis(extraction_results)
        
        if analysis_results:
            print("\n✅ Basic system test completed successfully!")
        else:
            print("\n❌ Analysis test failed")
    else:
        print("\n❌ Extraction test failed")

if __name__ == "__main__":
    main()
