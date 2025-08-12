"""
Complete System Test
Tests the entire PDF-to-Quiz pipeline from text extraction to question generation
"""

import os
import json
import time
import logging
from pathlib import Path
from text_extractor import TextExtractor
from content_analyzer import ContentAnalyzer
from question_generator import AIQuestionGenerator
from pdf_to_quiz_pipeline import PDFToQuizPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_text_extraction():
    """Test the text extraction component"""
    print("\n" + "="*60)
    print("🧪 TESTING TEXT EXTRACTION")
    print("="*60)
    
    try:
        extractor = TextExtractor()
        
        # Test with available PDFs
        pdf_files = [
            "Google IT Support Professional Certificate Notes (1).pdf",
            "CompTIA A+ Certification All-in-One Exam Guide , Tenth Edition (Exams 220-1001  220-1002) by Mike Meyers 3.pdf"
        ]
        
        extraction_results = {}
        
        for pdf_file in pdf_files:
            if os.path.exists(pdf_file):
                print(f"\n📄 Processing: {pdf_file}")
                
                start_time = time.time()
                results = extractor.extract_text(pdf_file, save_chunks=True)
                end_time = time.time()
                
                if 'error' not in results:
                    print(f"✅ Extraction successful!")
                    print(f"   📄 Total pages: {results.get('total_pages', 'Unknown')}")
                    print(f"   📝 Text length: {len(results.get('text', ''))} characters")
                    print(f"   🔧 Method: {results.get('method', 'Unknown')}")
                    print(f"   📊 Quality: {results.get('extraction_quality', 'Unknown')}")
                    print(f"   🧩 Chunks: {len(results.get('chunks', []))}")
                    print(f"   ⏱️  Time: {end_time - start_time:.2f} seconds")
                    
                    extraction_results[pdf_file] = results
                else:
                    print(f"❌ Extraction failed: {results['error']}")
            else:
                print(f"⚠️  PDF file not found: {pdf_file}")
        
        return extraction_results
        
    except Exception as e:
        print(f"❌ Text extraction test failed: {e}")
        return {}

def test_content_analysis(extraction_results):
    """Test the content analysis component"""
    print("\n" + "="*60)
    print("🧪 TESTING CONTENT ANALYSIS")
    print("="*60)
    
    try:
        analyzer = ContentAnalyzer()
        analysis_results = {}
        
        for pdf_file, extracted_data in extraction_results.items():
            print(f"\n📊 Analyzing content from: {pdf_file}")
            
            start_time = time.time()
            analysis = analyzer.analyze_content(extracted_data)
            end_time = time.time()
            
            if 'error' not in analysis:
                print(f"✅ Analysis successful!")
                print(f"   📊 Concepts found: {len(analysis.get('concepts', []))}")
                print(f"   📚 Topics identified: {len(analysis.get('topics', []))}")
                print(f"   📝 Facts extracted: {len(analysis.get('facts', []))}")
                print(f"   📋 Summary generated: {'Yes' if analysis.get('summary') else 'No'}")
                print(f"   ⏱️  Time: {end_time - start_time:.2f} seconds")
                
                # Save analysis results
                output_file = f"analysis_{Path(pdf_file).stem}.json"
                analyzer.save_analysis_results(analysis, output_file)
                print(f"   💾 Saved to: {output_file}")
                
                analysis_results[pdf_file] = analysis
            else:
                print(f"❌ Analysis failed: {analysis['error']}")
        
        return analysis_results
        
    except Exception as e:
        print(f"❌ Content analysis test failed: {e}")
        return {}

def test_question_generation(analysis_results):
    """Test the question generation component"""
    print("\n" + "="*60)
    print("🧪 TESTING QUESTION GENERATION")
    print("="*60)
    
    try:
        generator = AIQuestionGenerator(max_workers=2)  # Use 2 workers for testing
        question_results = {}
        
        for pdf_file, analysis_data in analysis_results.items():
            print(f"\n❓ Generating questions from: {pdf_file}")
            
            # Test with different difficulty distributions
            difficulty_distributions = [
                {'easy': 0.4, 'medium': 0.4, 'hard': 0.2},
                {'easy': 0.3, 'medium': 0.5, 'hard': 0.2}
            ]
            
            for i, difficulty_dist in enumerate(difficulty_distributions):
                print(f"   📈 Testing distribution {i+1}: {difficulty_dist}")
                
                start_time = time.time()
                questions = generator.generate_questions(
                    analysis_data,
                    questions_per_topic=2,  # Small number for testing
                    difficulty_distribution=difficulty_dist
                )
                end_time = time.time()
                
                if 'error' not in questions:
                    print(f"   ✅ Generation successful!")
                    print(f"      📝 Questions generated: {len(questions['questions'])}")
                    print(f"      📊 Topics covered: {questions['metadata']['topics_covered']}")
                    print(f"      🔤 Question types: {questions['metadata']['question_types']}")
                    print(f"      📈 Difficulty levels: {questions['metadata']['difficulty_levels']}")
                    print(f"      ⏱️  Time: {end_time - start_time:.2f} seconds")
                    
                    # Save questions
                    output_file = f"questions_{Path(pdf_file).stem}_dist{i+1}.json"
                    generator.save_questions(questions['questions'], output_file)
                    print(f"      💾 Saved to: {output_file}")
                    
                    question_results[f"{pdf_file}_dist{i+1}"] = questions
                else:
                    print(f"   ❌ Generation failed: {questions['error']}")
        
        return question_results
        
    except Exception as e:
        print(f"❌ Question generation test failed: {e}")
        return {}

def test_complete_pipeline():
    """Test the complete pipeline using the orchestration module"""
    print("\n" + "="*60)
    print("🧪 TESTING COMPLETE PIPELINE")
    print("="*60)
    
    try:
        pipeline = PDFToQuizPipeline()
        
        # Test with a single PDF for complete pipeline
        pdf_files = [
            "Google IT Support Professional Certificate Notes (1).pdf",
            "CompTIA A+ Certification All-in-One Exam Guide , Tenth Edition (Exams 220-1001  220-1002) by Mike Meyers 3.pdf"
        ]
        
        for pdf_file in pdf_files:
            if os.path.exists(pdf_file):
                print(f"\n🔄 Running complete pipeline for: {pdf_file}")
                
                start_time = time.time()
                results = pipeline.process_pdf(
                    pdf_file,
                    questions_per_topic=3,
                    difficulty_distribution={'easy': 0.4, 'medium': 0.4, 'hard': 0.2}
                )
                end_time = time.time()
                
                if 'error' not in results:
                    print(f"✅ Complete pipeline successful!")
                    print(f"   📄 PDF processed: {results['pdf_file']}")
                    print(f"   📝 Questions generated: {len(results['questions'])}")
                    print(f"   📊 Topics covered: {results['metadata']['topics_covered']}")
                    print(f"   ⏱️  Total time: {end_time - start_time:.2f} seconds")
                    
                    # Generate report
                    report = pipeline.generate_quiz_report(results)
                    print(f"   📋 Report generated: {len(report)} sections")
                else:
                    print(f"❌ Complete pipeline failed: {results['error']}")
            else:
                print(f"⚠️  PDF file not found: {pdf_file}")
        
    except Exception as e:
        print(f"❌ Complete pipeline test failed: {e}")

def analyze_test_results(extraction_results, analysis_results, question_results):
    """Analyze and summarize test results"""
    print("\n" + "="*60)
    print("📊 TEST RESULTS SUMMARY")
    print("="*60)
    
    # Extraction results
    print(f"\n📄 Text Extraction:")
    total_extracted = len(extraction_results)
    successful_extractions = sum(1 for r in extraction_results.values() if 'error' not in r)
    print(f"   Files processed: {total_extracted}")
    print(f"   Successful extractions: {successful_extractions}")
    print(f"   Success rate: {(successful_extractions/total_extracted)*100:.1f}%" if total_extracted > 0 else "N/A")
    
    # Analysis results
    print(f"\n📊 Content Analysis:")
    total_analyzed = len(analysis_results)
    successful_analyses = sum(1 for r in analysis_results.values() if 'error' not in r)
    print(f"   Files analyzed: {total_analyzed}")
    print(f"   Successful analyses: {successful_analyses}")
    print(f"   Success rate: {(successful_analyses/total_analyzed)*100:.1f}%" if total_analyzed > 0 else "N/A")
    
    # Question generation results
    print(f"\n❓ Question Generation:")
    total_generated = len(question_results)
    successful_generations = sum(1 for r in question_results.values() if 'error' not in r)
    total_questions = sum(len(r['questions']) for r in question_results.values() if 'error' not in r)
    print(f"   Generation attempts: {total_generated}")
    print(f"   Successful generations: {successful_generations}")
    print(f"   Total questions generated: {total_questions}")
    print(f"   Success rate: {(successful_generations/total_generated)*100:.1f}%" if total_generated > 0 else "N/A")
    
    # Performance metrics
    print(f"\n⚡ Performance Metrics:")
    if question_results:
        generator = AIQuestionGenerator()
        metrics = generator.get_performance_metrics()
        print(f"   Model used: {metrics['model_used']}")
        print(f"   Parallel workers: {metrics['parallel_workers']}")
        print(f"   Total questions in cache: {metrics['total_questions_generated']}")
    
    # File outputs
    print(f"\n💾 Generated Files:")
    output_files = [f for f in os.listdir('.') if f.endswith('.json') and ('extracted_' in f or 'analysis_' in f or 'questions_' in f)]
    for file in sorted(output_files):
        file_size = os.path.getsize(file) / 1024  # KB
        print(f"   {file} ({file_size:.1f} KB)")

def main():
    """Run the complete system test"""
    print("🚀 STARTING COMPLETE SYSTEM TEST")
    print("="*60)
    
    start_time = time.time()
    
    try:
        # Test individual components
        extraction_results = test_text_extraction()
        analysis_results = test_content_analysis(extraction_results)
        question_results = test_question_generation(analysis_results)
        
        # Test complete pipeline
        test_complete_pipeline()
        
        # Analyze results
        analyze_test_results(extraction_results, analysis_results, question_results)
        
        end_time = time.time()
        print(f"\n✅ Complete system test finished in {end_time - start_time:.2f} seconds")
        
    except Exception as e:
        print(f"\n❌ System test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
