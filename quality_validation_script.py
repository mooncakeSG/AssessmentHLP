"""
Quality Validation Script for Cross-Reference Question Generator
Helps validate question quality and provides enhancement suggestions
"""

import json
import os
from typing import Dict, List, Any

def analyze_question_quality(questions_file: str) -> Dict[str, Any]:
    """Analyze the quality of generated questions"""
    print(f"🔍 Analyzing question quality from: {questions_file}")
    
    with open(questions_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    questions = data.get('questions', [])
    
    # Quality metrics
    quality_metrics = {
        'total_questions': len(questions),
        'question_types': {},
        'avg_question_length': 0,
        'avg_answer_length': 0,
        'questions_with_answers': 0,
        'questions_without_answers': 0,
        'technical_terms_found': 0,
        'quality_score': 0
    }
    
    technical_terms = [
        'SSL', 'TLS', 'HTTPS', 'authentication', 'encryption', 'firewall',
        'VPN', 'DNS', 'DHCP', 'TCP/IP', 'router', 'switch', 'network',
        'security', 'password', 'protocol', 'certificate', 'encryption'
    ]
    
    total_question_length = 0
    total_answer_length = 0
    
    for question in questions:
        # Question type analysis
        q_type = question.get('type', 'unknown')
        quality_metrics['question_types'][q_type] = quality_metrics['question_types'].get(q_type, 0) + 1
        
        # Length analysis
        question_text = question.get('question', '')
        answer_text = question.get('answer', '')
        
        total_question_length += len(question_text)
        total_answer_length += len(answer_text)
        
        # Answer completeness
        if answer_text.strip():
            quality_metrics['questions_with_answers'] += 1
        else:
            quality_metrics['questions_without_answers'] += 1
        
        # Technical term detection
        question_lower = question_text.lower()
        for term in technical_terms:
            if term.lower() in question_lower:
                quality_metrics['technical_terms_found'] += 1
                break
    
    # Calculate averages
    if questions:
        quality_metrics['avg_question_length'] = total_question_length / len(questions)
        quality_metrics['avg_answer_length'] = total_answer_length / len(questions)
    
    # Calculate quality score
    completeness_score = quality_metrics['questions_with_answers'] / quality_metrics['total_questions'] if quality_metrics['total_questions'] > 0 else 0
    technical_score = quality_metrics['technical_terms_found'] / quality_metrics['total_questions'] if quality_metrics['total_questions'] > 0 else 0
    length_score = min(quality_metrics['avg_question_length'] / 100, 1.0)  # Normalize to 0-1
    
    quality_metrics['quality_score'] = (completeness_score * 0.4 + technical_score * 0.4 + length_score * 0.2) * 100
    
    return quality_metrics

def generate_enhancement_suggestions(quality_metrics: Dict[str, Any]) -> List[str]:
    """Generate suggestions for improving question quality"""
    suggestions = []
    
    # Answer completeness
    if quality_metrics['questions_without_answers'] > 0:
        suggestions.append(f"⚠️  {quality_metrics['questions_without_answers']} questions lack answers - consider regenerating")
    
    # Question type diversity
    if len(quality_metrics['question_types']) < 2:
        suggestions.append("📝 Consider generating more diverse question types (multiple choice, true/false, short answer)")
    
    # Technical depth
    if quality_metrics['technical_terms_found'] < quality_metrics['total_questions'] * 0.7:
        suggestions.append("🔧 Questions could benefit from more technical terminology")
    
    # Question length
    if quality_metrics['avg_question_length'] < 50:
        suggestions.append("📏 Questions are quite short - consider more detailed questions")
    elif quality_metrics['avg_question_length'] > 200:
        suggestions.append("📏 Questions are very long - consider more concise questions")
    
    # Overall quality
    if quality_metrics['quality_score'] < 70:
        suggestions.append("🎯 Overall quality score is low - review generation parameters")
    elif quality_metrics['quality_score'] > 90:
        suggestions.append("🌟 Excellent quality score! System is working well")
    
    return suggestions

def main():
    """Main validation function"""
    print("🔍 Cross-Reference Question Generator - Quality Validation")
    print("="*60)
    
    # Find the most recent questions file (prioritize improved questions)
    questions_files = []
    
    # Look for improved questions first
    improved_files = [f for f in os.listdir('.') if f.startswith('improved_questions_') and f.endswith('.json')]
    if improved_files:
        questions_files.extend(improved_files)
    
    # Then look for web-generated questions
    web_files = [f for f in os.listdir('.') if f.startswith('web_generated_questions_') and f.endswith('.json')]
    if web_files:
        questions_files.extend(web_files)
    
    if not questions_files:
        print("❌ No generated questions files found")
        return
    
    # Use the most recent file
    latest_file = max(questions_files, key=os.path.getctime)
    print(f"📄 Analyzing: {latest_file}")
    
    # Analyze quality
    quality_metrics = analyze_question_quality(latest_file)
    
    # Display results
    print(f"\n📊 Quality Analysis Results:")
    print(f"   Total Questions: {quality_metrics['total_questions']}")
    print(f"   Questions with Answers: {quality_metrics['questions_with_answers']}")
    print(f"   Questions without Answers: {quality_metrics['questions_without_answers']}")
    print(f"   Average Question Length: {quality_metrics['avg_question_length']:.1f} characters")
    print(f"   Average Answer Length: {quality_metrics['avg_answer_length']:.1f} characters")
    print(f"   Technical Terms Found: {quality_metrics['technical_terms_found']}")
    print(f"   Quality Score: {quality_metrics['quality_score']:.1f}/100")
    
    print(f"\n📋 Question Type Distribution:")
    for q_type, count in quality_metrics['question_types'].items():
        print(f"   {q_type.replace('_', ' ').title()}: {count}")
    
    # Generate suggestions
    suggestions = generate_enhancement_suggestions(quality_metrics)
    
    print(f"\n💡 Enhancement Suggestions:")
    for suggestion in suggestions:
        print(f"   {suggestion}")
    
    # Quality assessment
    print(f"\n🎯 Quality Assessment:")
    if quality_metrics['quality_score'] >= 90:
        print("   🌟 EXCELLENT - System is generating high-quality questions")
    elif quality_metrics['quality_score'] >= 80:
        print("   ✅ GOOD - Questions are generally well-formed")
    elif quality_metrics['quality_score'] >= 70:
        print("   ⚠️  FAIR - Some improvements needed")
    else:
        print("   ❌ NEEDS IMPROVEMENT - Review generation parameters")

if __name__ == "__main__":
    main()
