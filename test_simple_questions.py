"""
Test simple question generation with relaxed validation
"""

import os
import json
from question_generator import AIQuestionGenerator

def test_simple_question_generation():
    print("🧪 Testing simple question generation...")
    
    # Load analysis results
    analysis_file = "analysis_test.json"
    if not os.path.exists(analysis_file):
        print(f"❌ Analysis file not found: {analysis_file}")
        return None
    
    print(f"📂 Loading analysis from: {analysis_file}")
    with open(analysis_file, 'r', encoding='utf-8') as f:
        analysis_data = json.load(f)
    
    # Initialize question generator with relaxed settings
    generator = AIQuestionGenerator(max_workers=1)
    
    print("🔄 Generating questions with relaxed validation...")
    try:
        # Generate questions with minimal validation
        questions = generator.generate_questions(
            analysis_data,
            questions_per_topic=1,  # Just 1 question per topic
            difficulty_distribution={'easy': 1.0}  # All easy questions
        )
        
        if 'error' not in questions:
            print(f"✅ Question generation successful!")
            print(f"   📝 Questions generated: {len(questions['questions'])}")
            
            # Save raw questions before validation
            raw_questions = questions.get('raw_questions', [])
            if raw_questions:
                print(f"   📝 Raw questions before validation: {len(raw_questions)}")
                
                # Save raw questions
                with open("raw_questions_test.json", 'w', encoding='utf-8') as f:
                    json.dump(raw_questions, f, indent=2, ensure_ascii=False)
                print(f"   💾 Raw questions saved to: raw_questions_test.json")
                
                # Show a sample raw question
                sample_q = raw_questions[0]
                print(f"\n📋 Sample Raw Question:")
                print(f"   Question: {sample_q.get('question', 'No question text')}")
                if 'options' in sample_q:
                    print(f"   Options: {sample_q['options']}")
                print(f"   Answer: {sample_q.get('answer', 'No answer')}")
            
            # Save final questions
            output_file = "questions_simple_test.json"
            generator.save_questions(questions['questions'], output_file)
            print(f"   💾 Final questions saved to: {output_file}")
            
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
    print("🚀 Testing Simple Question Generation")
    print("="*50)
    
    results = test_simple_question_generation()
    
    if results:
        print("\n✅ Simple question generation test completed!")
    else:
        print("\n❌ Simple question generation test failed")

if __name__ == "__main__":
    main()
