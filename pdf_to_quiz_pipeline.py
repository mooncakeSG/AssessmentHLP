"""
PDF-to-Quiz Pipeline
Main orchestration module that integrates all components into a complete workflow
"""

import os
import json
import logging
import time
from typing import Dict, List, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

# Import our modules
from text_extractor import TextExtractor
from content_analyzer import ContentAnalyzer
from question_generator import AIQuestionGenerator

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PDFToQuizPipeline:
    """
    Complete pipeline for converting PDF documents to quiz questions
    """
    
    def __init__(self, output_dir: str = "output"):
        """
        Initialize the pipeline
        
        Args:
            output_dir: Directory to save output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.text_extractor = TextExtractor()
        self.content_analyzer = ContentAnalyzer()
        self.question_generator = AIQuestionGenerator()
        
        logger.info("✅ PDF-to-Quiz Pipeline initialized successfully")
    
    def process_pdf(self, pdf_path: str, 
                   questions_per_topic: int = 10,
                   question_types: List[str] = None,
                   save_intermediate: bool = True) -> Dict[str, Any]:
        """
        Process a PDF file through the complete pipeline
        
        Args:
            pdf_path: Path to PDF file
            questions_per_topic: Number of questions to generate per topic
            question_types: Types of questions to generate
            save_intermediate: Whether to save intermediate results
            
        Returns:
            Complete pipeline results
        """
        start_time = time.time()
        pdf_name = Path(pdf_path).stem
        
        logger.info(f"🚀 Starting PDF-to-Quiz pipeline for: {pdf_name}")
        
        try:
            # Step 1: Text Extraction
            logger.info("📄 Step 1: Extracting text from PDF...")
            extraction_results = self.text_extractor.extract_text(pdf_path)
            
            if 'error' in extraction_results:
                return {'error': f"Text extraction failed: {extraction_results['error']}"}
            
            extraction_time = time.time() - start_time
            logger.info(f"✅ Text extraction completed in {extraction_time:.2f}s")
            
            # Save extraction results
            if save_intermediate:
                extraction_file = self.output_dir / f"{pdf_name}_extraction.json"
                self._save_json(extraction_results, extraction_file)
            
            # Step 2: Content Analysis
            logger.info("🔍 Step 2: Analyzing content...")
            analysis_start = time.time()
            analysis_results = self.content_analyzer.analyze_content(extraction_results)
            
            if 'error' in analysis_results:
                return {'error': f"Content analysis failed: {analysis_results['error']}"}
            
            analysis_time = time.time() - analysis_start
            logger.info(f"✅ Content analysis completed in {analysis_time:.2f}s")
            
            # Save analysis results
            if save_intermediate:
                analysis_file = self.output_dir / f"{pdf_name}_analysis.json"
                self.content_analyzer.save_analysis_results(analysis_results, str(analysis_file))
            
            # Step 3: Question Generation
            logger.info("❓ Step 3: Generating questions...")
            generation_start = time.time()
            
            if question_types is None:
                question_types = ['multiple_choice', 'true_false', 'short_answer']
            
            question_results = self.question_generator.generate_questions(
                analysis_results, questions_per_topic, question_types
            )
            
            if 'error' in question_results:
                return {'error': f"Question generation failed: {question_results['error']}"}
            
            generation_time = time.time() - generation_start
            logger.info(f"✅ Question generation completed in {generation_time:.2f}s")
            
            # Save question results
            questions_file = self.output_dir / f"{pdf_name}_questions.json"
            self.question_generator.save_questions(question_results['questions'], str(questions_file))
            
            # Calculate total time
            total_time = time.time() - start_time
            
            # Create comprehensive results
            pipeline_results = {
                'pdf_name': pdf_name,
                'pdf_path': pdf_path,
                'extraction': {
                    'method': extraction_results.get('method', 'unknown'),
                    'total_pages': extraction_results.get('total_pages', 0),
                    'text_length': len(extraction_results.get('text', '')),
                    'chunks': len(extraction_results.get('chunks', [])),
                    'quality': extraction_results.get('extraction_quality', 'unknown'),
                    'time': extraction_time
                },
                'analysis': {
                    'concepts': len(analysis_results.get('concepts', [])),
                    'topics': len(analysis_results.get('topics', [])),
                    'facts': len(analysis_results.get('facts', [])),
                    'time': analysis_time
                },
                'questions': {
                    'total': question_results['metadata']['total_questions'],
                    'topics_covered': question_results['metadata']['topics_covered'],
                    'concepts_covered': question_results['metadata']['concepts_covered'],
                    'types': question_results['metadata']['question_types'],
                    'quality': question_results['metadata']['generation_quality'],
                    'time': generation_time
                },
                'performance': {
                    'total_time': total_time,
                    'pages_per_minute': (extraction_results.get('total_pages', 0) / total_time) * 60,
                    'questions_per_minute': (question_results['metadata']['total_questions'] / total_time) * 60
                },
                'files': {
                    'extraction': str(extraction_file) if save_intermediate else None,
                    'analysis': str(analysis_file) if save_intermediate else None,
                    'questions': str(questions_file)
                }
            }
            
            # Save pipeline results
            pipeline_file = self.output_dir / f"{pdf_name}_pipeline_results.json"
            self._save_json(pipeline_results, pipeline_file)
            
            logger.info(f"🎉 Pipeline completed successfully in {total_time:.2f}s")
            logger.info(f"📊 Generated {question_results['metadata']['total_questions']} questions")
            logger.info(f"📁 Results saved to: {self.output_dir}")
            
            return pipeline_results
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            return {'error': str(e)}
    
    def process_multiple_pdfs(self, pdf_paths: List[str], 
                            questions_per_topic: int = 10,
                            question_types: List[str] = None,
                            parallel: bool = False) -> Dict[str, Any]:
        """
        Process multiple PDF files
        
        Args:
            pdf_paths: List of PDF file paths
            questions_per_topic: Number of questions per topic
            question_types: Types of questions to generate
            parallel: Whether to process in parallel (future enhancement)
            
        Returns:
            Results for all PDFs
        """
        logger.info(f"🔄 Processing {len(pdf_paths)} PDF files...")
        
        all_results = {}
        successful = 0
        failed = 0
        
        for i, pdf_path in enumerate(pdf_paths, 1):
            logger.info(f"📄 Processing PDF {i}/{len(pdf_paths)}: {Path(pdf_path).name}")
            
            try:
                result = self.process_pdf(pdf_path, questions_per_topic, question_types)
                
                if 'error' not in result:
                    all_results[Path(pdf_path).name] = result
                    successful += 1
                else:
                    all_results[Path(pdf_path).name] = {'error': result['error']}
                    failed += 1
                    
            except Exception as e:
                logger.error(f"❌ Failed to process {pdf_path}: {e}")
                all_results[Path(pdf_path).name] = {'error': str(e)}
                failed += 1
        
        # Create summary
        summary = {
            'total_pdfs': len(pdf_paths),
            'successful': successful,
            'failed': failed,
            'total_questions': sum(r.get('questions', {}).get('total', 0) 
                                 for r in all_results.values() if 'error' not in r),
            'total_time': sum(r.get('performance', {}).get('total_time', 0) 
                            for r in all_results.values() if 'error' not in r),
            'results': all_results
        }
        
        # Save summary
        summary_file = self.output_dir / "batch_processing_summary.json"
        self._save_json(summary, summary_file)
        
        logger.info(f"🎉 Batch processing completed: {successful} successful, {failed} failed")
        return summary
    
    def generate_quiz_report(self, pipeline_results: Dict[str, Any]) -> str:
        """
        Generate a human-readable report from pipeline results
        
        Args:
            pipeline_results: Results from pipeline processing
            
        Returns:
            Formatted report string
        """
        if 'error' in pipeline_results:
            return f"❌ Pipeline Error: {pipeline_results['error']}"
        
        report = f"""
📊 PDF-to-Quiz Pipeline Report
{'='*50}

📄 Document: {pipeline_results['pdf_name']}
📁 Path: {pipeline_results['pdf_path']}

📈 EXTRACTION RESULTS
• Method: {pipeline_results['extraction']['method']}
• Pages: {pipeline_results['extraction']['total_pages']}
• Text Length: {pipeline_results['extraction']['text_length']:,} characters
• Chunks: {pipeline_results['extraction']['chunks']}
• Quality: {pipeline_results['extraction']['quality']}
• Time: {pipeline_results['extraction']['time']:.2f}s

🔍 ANALYSIS RESULTS
• Concepts: {pipeline_results['analysis']['concepts']}
• Topics: {pipeline_results['analysis']['topics']}
• Facts: {pipeline_results['analysis']['facts']}
• Time: {pipeline_results['analysis']['time']:.2f}s

❓ QUESTION GENERATION
• Total Questions: {pipeline_results['questions']['total']}
• Topics Covered: {pipeline_results['questions']['topics_covered']}
• Concepts Covered: {pipeline_results['questions']['concepts_covered']}
• Question Types: {', '.join(pipeline_results['questions']['types'])}
• Quality: {pipeline_results['questions']['quality']}
• Time: {pipeline_results['questions']['time']:.2f}s

⚡ PERFORMANCE METRICS
• Total Time: {pipeline_results['performance']['total_time']:.2f}s
• Pages/Minute: {pipeline_results['performance']['pages_per_minute']:.1f}
• Questions/Minute: {pipeline_results['performance']['questions_per_minute']:.1f}

📁 OUTPUT FILES
• Extraction: {pipeline_results['files']['extraction'] or 'Not saved'}
• Analysis: {pipeline_results['files']['analysis'] or 'Not saved'}
• Questions: {pipeline_results['files']['questions']}

🎯 SUMMARY
✅ Successfully processed {pipeline_results['extraction']['total_pages']} pages
✅ Generated {pipeline_results['questions']['total']} high-quality questions
✅ Covered {pipeline_results['questions']['topics_covered']} topics
✅ Achieved {pipeline_results['questions']['quality']} quality level
"""
        
        return report
    
    def _save_json(self, data: Dict[str, Any], file_path: Path):
        """
        Save data to JSON file
        
        Args:
            data: Data to save
            file_path: File path
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 Saved to: {file_path}")
        except Exception as e:
            logger.error(f"❌ Error saving {file_path}: {e}")

def main():
    """Test the complete pipeline"""
    # Initialize pipeline
    pipeline = PDFToQuizPipeline()
    
    # Test with available PDFs
    pdf_files = [
        "Google IT Support Professional Certificate Notes (1).pdf",
        "CompTIA A+ Certification All-in-One Exam Guide , Tenth Edition (Exams 220-1001  220-1002) by Mike Meyers 3.pdf"
    ]
    
    # Filter to existing files
    existing_pdfs = [pdf for pdf in pdf_files if os.path.exists(pdf)]
    
    if not existing_pdfs:
        print("❌ No PDF files found for testing")
        return
    
    print(f"🔍 Found {len(existing_pdfs)} PDF files for testing")
    
    # Process first PDF as a test
    test_pdf = existing_pdfs[0]
    print(f"\n🚀 Testing pipeline with: {test_pdf}")
    
    try:
        # Process the PDF
        results = pipeline.process_pdf(
            test_pdf,
            questions_per_topic=5,  # Reduced for testing
            question_types=['multiple_choice', 'true_false'],
            save_intermediate=True
        )
        
        if 'error' not in results:
            # Generate and display report
            report = pipeline.generate_quiz_report(results)
            print(report)
            
            # Show sample questions
            questions_file = results['files']['questions']
            if os.path.exists(questions_file):
                with open(questions_file, 'r', encoding='utf-8') as f:
                    questions_data = json.load(f)
                
                print(f"\n📝 Sample Questions (showing first 3):")
                for i, question in enumerate(questions_data[:3]):
                    print(f"\nQuestion {i+1}:")
                    print(f"  Type: {question['type']}")
                    print(f"  Difficulty: {question['difficulty']}")
                    print(f"  Topic: {question['topic']}")
                    print(f"  Question: {question['question']}")
                    if question['type'] == 'multiple_choice':
                        print(f"  Options: {question['options']}")
                    print(f"  Answer: {question['correct_answer']}")
                    print(f"  Explanation: {question['explanation']}")
        else:
            print(f"❌ Pipeline failed: {results['error']}")
            
    except Exception as e:
        print(f"❌ Error running pipeline: {e}")

if __name__ == "__main__":
    main()
