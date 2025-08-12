"""
Test Cross-Reference Question Generator with Real CompTIA Content
Extracts content from CompTIA PDF and generates questions using Google notes
"""

import os
import json
import fitz  # PyMuPDF
from cross_reference_question_generator import CrossReferenceQuestionGenerator

def extract_comptia_chapters():
    """Extract specific chapters from CompTIA PDF"""
    print("📚 Extracting CompTIA Content")
    print("="*50)
    
    pdf_file = "CompTIA A+ Certification All-in-One Exam Guide , Tenth Edition (Exams 220-1001  220-1002) by Mike Meyers 3.pdf"
    
    if not os.path.exists(pdf_file):
        print(f"❌ CompTIA PDF not found: {pdf_file}")
        return None
    
    try:
        doc = fitz.open(pdf_file)
        print(f"📄 Opened CompTIA PDF: {len(doc)} pages")
        
        # Define chapter ranges (approximate page ranges)
        chapters = {
            "networking_fundamentals": {
                "name": "Networking Fundamentals",
                "start_page": 400,
                "end_page": 500,
                "description": "Network protocols, TCP/IP, DNS, DHCP"
            },
            "security_basics": {
                "name": "Security Basics", 
                "start_page": 600,
                "end_page": 700,
                "description": "Firewalls, VPNs, authentication, encryption"
            },
            "hardware_troubleshooting": {
                "name": "Hardware Troubleshooting",
                "start_page": 200,
                "end_page": 300,
                "description": "RAM, CPU, storage, power supply issues"
            },
            "operating_systems": {
                "name": "Operating Systems",
                "start_page": 300,
                "end_page": 400,
                "description": "Windows, Linux, macOS, virtualization"
            }
        }
        
        extracted_chapters = {}
        
        for chapter_id, chapter_info in chapters.items():
            print(f"\n📖 Extracting: {chapter_info['name']}")
            print(f"   Pages: {chapter_info['start_page']}-{chapter_info['end_page']}")
            
            text = ""
            for page_num in range(chapter_info['start_page'], min(chapter_info['end_page'], len(doc))):
                page = doc[page_num]
                text += page.get_text()
            
            # Clean and segment the text
            text = clean_text(text)
            
            if len(text) > 1000:  # Only keep substantial content
                extracted_chapters[chapter_id] = {
                    "name": chapter_info['name'],
                    "description": chapter_info['description'],
                    "text": text,
                    "length": len(text),
                    "pages": f"{chapter_info['start_page']}-{chapter_info['end_page']}"
                }
                print(f"   ✅ Extracted {len(text)} characters")
            else:
                print(f"   ⚠️  Content too short ({len(text)} chars), skipping")
        
        doc.close()
        return extracted_chapters
        
    except Exception as e:
        print(f"❌ Error extracting CompTIA content: {e}")
        return None

def clean_text(text):
    """Clean extracted text"""
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    # Remove page numbers and headers
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        # Skip lines that are just numbers (page numbers)
        if line.isdigit() and len(line) < 4:
            continue
        # Skip very short lines that might be headers
        if len(line) < 3:
            continue
        cleaned_lines.append(line)
    
    return ' '.join(cleaned_lines)

def test_chapter_questions(chapter_id, chapter_data, generator):
    """Test question generation for a specific chapter"""
    print(f"\n🧪 Testing: {chapter_data['name']}")
    print(f"📝 Content length: {chapter_data['length']} characters")
    print(f"📄 Pages: {chapter_data['pages']}")
    
    try:
        # Generate questions
        results = generator.generate_cross_reference_questions(
            comptia_text=chapter_data['text'],
            num_questions=5,
            question_types=['multiple_choice', 'true_false', 'short_answer']
        )
        
        # Display results
        print(f"✅ Generated {len(results['questions'])} questions")
        print(f"📊 Concepts extracted: {results['num_concepts_extracted']}")
        print(f"📄 Google notes chunks retrieved: {results['num_chunks_retrieved']}")
        
        # Show sample questions
        print(f"\n📋 Sample Questions:")
        for i, question in enumerate(results['questions'][:3], 1):  # Show first 3
            print(f"\nQ{i} ({question['type'].replace('_', ' ').title()}):")
            print(f"   {question['question']}")
            print(f"   Answer: {question['answer']}")
        
        # Show top concepts
        print(f"\n🔍 Top Concepts from {chapter_data['name']}:")
        concepts = results['concept_analysis']['concepts']
        for i, concept in enumerate(concepts[:5], 1):
            print(f"   {i}. {concept['text']} ({concept['type']})")
        
        # Save results
        output_file = f"comptia_{chapter_id}_questions_{int(__import__('time').time())}.json"
        generator.save_results(results, output_file)
        print(f"\n💾 Results saved to: {output_file}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error generating questions for {chapter_id}: {e}")
        return None

def main():
    """Main test function"""
    print("🚀 Testing Cross-Reference Question Generator with Real CompTIA Content")
    print("="*80)
    
    # Check if Google notes exist
    google_notes_file = "extracted_Google IT Support Professional Certificate Notes (1).json"
    if not os.path.exists(google_notes_file):
        print(f"❌ Google notes file not found: {google_notes_file}")
        print("Please run extract_google_notes_only.py first")
        return
    
    print(f"✅ Found Google notes: {google_notes_file}")
    
    # Initialize generator
    print("\n🔄 Initializing Cross-Reference Question Generator...")
    generator = CrossReferenceQuestionGenerator()
    
    # Extract CompTIA chapters
    chapters = extract_comptia_chapters()
    if not chapters:
        print("❌ Failed to extract CompTIA content")
        return
    
    print(f"\n📚 Extracted {len(chapters)} chapters from CompTIA PDF")
    
    # Test each chapter
    all_results = {}
    
    for chapter_id, chapter_data in chapters.items():
        print(f"\n" + "="*60)
        results = test_chapter_questions(chapter_id, chapter_data, generator)
        if results:
            all_results[chapter_id] = results
    
    # Summary
    print(f"\n" + "="*80)
    print("🎉 COMPTIA CONTENT TESTING COMPLETED")
    print("="*80)
    
    print(f"\n📊 Summary:")
    print(f"   Chapters tested: {len(all_results)}")
    total_questions = sum(len(results['questions']) for results in all_results.values())
    print(f"   Total questions generated: {total_questions}")
    
    print(f"\n📋 Chapter Results:")
    for chapter_id, results in all_results.items():
        chapter_name = chapters[chapter_id]['name']
        num_questions = len(results['questions'])
        num_concepts = results['num_concepts_extracted']
        print(f"   {chapter_name}: {num_questions} questions, {num_concepts} concepts")
    
    print(f"\n💡 Next Steps:")
    print(f"   1. Review generated questions for quality")
    print(f"   2. Create web interface for easy interaction")
    print(f"   3. Add export functionality for quiz platforms")
    print(f"   4. Implement analytics and quality metrics")
    
    # Save comprehensive results
    comprehensive_results = {
        "test_summary": {
            "chapters_tested": len(all_results),
            "total_questions": total_questions,
            "timestamp": __import__('time').time()
        },
        "chapter_results": all_results,
        "chapter_metadata": chapters
    }
    
    output_file = f"comptia_comprehensive_test_{int(__import__('time').time())}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(comprehensive_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Comprehensive results saved to: {output_file}")

if __name__ == "__main__":
    main()
