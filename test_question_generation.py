"""
Test question generation component
"""

import os
import json
from question_generator import AIQuestionGenerator

def test_question_generation():
    print("🧪 Testing question generation...")
    
    # Load analysis results
    analysis_file = "analysis_test.json"
    if not os.path.exists(analysis_file):
        print(f"❌ Analysis file not found: {analysis_file}")
        return None
    
    print(f"📂 Loading analysis from: {analysis_file}")
    with open(analysis_file, 'r', encoding='utf-8') as f:
        analysis_data = json.load(f)
    
    # Initialize question generator
    generator = AIQuestionGenerator(max_workers=1)  # Use 1 worker for testing
    
    print("🔄 Generating questions...")
    try:
        questions = generator.generate_questions(
            analysis_data,
            questions_per_topic=2,  # Small number for testing
            difficulty_distribution={'easy': 0.5, 'medium': 0.3, 'hard': 0.2}
        )
        
        if 'error' not in questions:
            print(f"✅ Question generation successful!")
            print(f"   📝 Questions generated: {len(questions['questions'])}")
            print(f"   📊 Topics covered: {questions['metadata']['topics_covered']}")
            print(f"   🔤 Question types: {questions['metadata']['question_types']}")
            print(f"   📈 Difficulty levels: {questions['metadata']['difficulty_levels']}")
            
            # Save questions
            output_file = "questions_test.json"
            generator.save_questions(questions['questions'], output_file)
            print(f"   💾 Saved to: {output_file}")
            
            # Show a sample question
            if questions['questions']:
                sample_q = questions['questions'][0]
                print(f"\n📋 Sample Question:")
                print(f"   Type: {sample_q.get('type', 'Unknown')}")
                print(f"   Difficulty: {sample_q.get('difficulty', 'Unknown')}")
                print(f"   Question: {sample_q.get('question', 'No question text')}")
                if 'options' in sample_q:
                    print(f"   Options: {len(sample_q['options'])} choices")
                print(f"   Answer: {sample_q.get('answer', 'No answer')}")
            
            return questions
        else:
            print(f"❌ Question generation failed: {questions['error']}")
            return None
            
    except Exception as e:
        print(f"❌ Error during question generation: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("🚀 Testing Question Generation Component")
    print("="*50)
    
    results = test_question_generation()
    
    if results:
        print("\n✅ Question generation test completed successfully!")
    else:
        print("\n❌ Question generation test failed")

if __name__ == "__main__":
    main()
