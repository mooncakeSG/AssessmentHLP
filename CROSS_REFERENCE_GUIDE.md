# Cross-Reference Question Generator Guide

## Overview

The Cross-Reference Question Generator is a sophisticated system that generates highly accurate quiz questions by:

1. **Extracting concepts** from CompTIA textbook content using NLP techniques
2. **Matching concepts** with relevant Google IT Support notes using semantic similarity
3. **Generating questions** based strictly on the Google notes content using AI

This ensures that all questions are grounded in the Google IT Support curriculum while being relevant to CompTIA topics.

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  CompTIA Text   │───▶│  Concept        │───▶│  Google Notes   │
│  (Input)        │    │  Extraction     │    │  Matching       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Generated      │◀───│  AI Question    │◀───│  Relevant       │
│  Questions      │    │  Generation     │    │  Chunks         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Key Features

### ✅ **Concept Extraction**
- **NLP Processing**: Uses spaCy for named entity recognition and noun phrase extraction
- **IT Terminology**: Recognizes technical terms across networking, hardware, security, etc.
- **Weighted Ranking**: Prioritizes entities and technical terms over general concepts

### ✅ **Semantic Matching**
- **Embedding Similarity**: Uses Sentence-BERT to find relevant Google notes chunks
- **Top-K Retrieval**: Returns the most relevant chunks for question generation
- **Similarity Scoring**: Provides confidence scores for each matched chunk

### ✅ **AI Question Generation**
- **Strict Source Control**: Questions are based ONLY on Google notes content
- **Multiple Formats**: Supports multiple choice, true/false, and short answer questions
- **Context Preservation**: Maintains relevance to original CompTIA concepts

## Installation & Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Set Up Environment
Create a `.env` file with your Groq API key:
```
GROQ_API_KEY=your_api_key_here
```

### 3. Extract Google Notes
```bash
python extract_google_notes_only.py
```

## Usage Examples

### Basic Usage

```python
from cross_reference_question_generator import CrossReferenceQuestionGenerator

# Initialize the generator
generator = CrossReferenceQuestionGenerator()

# CompTIA content (your input)
comptia_text = """
Network Security Fundamentals

Network security is a critical component of modern IT infrastructure. 
Firewalls act as the first line of defense, filtering traffic based on 
predefined rules. VPNs provide secure remote access by encrypting data transmission.
"""

# Generate questions
results = generator.generate_cross_reference_questions(
    comptia_text=comptia_text,
    num_questions=5,
    question_types=['multiple_choice', 'true_false', 'short_answer']
)

# Access results
print(f"Generated {len(results['questions'])} questions")
for question in results['questions']:
    print(f"Q: {question['question']}")
    print(f"A: {question['answer']}")
```

### Advanced Usage

```python
# Custom configuration
generator = CrossReferenceQuestionGenerator(
    google_notes_path="path/to/your/notes.json",
    max_workers=4
)

# Generate specific question types
results = generator.generate_cross_reference_questions(
    comptia_text=comptia_text,
    num_questions=10,
    question_types=['multiple_choice']  # Only multiple choice
)

# Save results
output_file = generator.save_results(results, "my_questions.json")
```

## Demo Scripts

### 1. Complete Demo
```bash
python demo_cross_reference.py
```
Shows the full system in action with sample CompTIA content.

### 2. Test System
```bash
python test_cross_reference.py
```
Comprehensive testing of all system components.

## Output Format

The system generates structured JSON output:

```json
{
  "questions": [
    {
      "type": "multiple_choice",
      "question": "Which of the following best describes...",
      "answer": "B) Correct answer with explanation",
      "source": "google_notes"
    }
  ],
  "concept_analysis": {
    "concepts": [
      {
        "text": "Firewall",
        "type": "it_term",
        "weight": 3,
        "source": "it_terminology"
      }
    ],
    "entities": [...],
    "noun_phrases": [...],
    "technical_terms": [...],
    "it_terms": [...]
  },
  "relevant_chunks": [
    {
      "chunk": {
        "id": "0_123",
        "text": "Google notes content...",
        "embedding": [...]
      },
      "similarity": 0.85,
      "score": 0.85
    }
  ],
  "metadata": {
    "comptia_text_length": 1023,
    "num_concepts_extracted": 49,
    "num_chunks_retrieved": 10
  }
}
```

## Key Benefits

### 🎯 **Accuracy**
- Questions are based on verified Google IT Support content
- No hallucination or incorrect information
- Strict adherence to source material

### 🔍 **Relevance**
- Semantic matching ensures topic alignment
- Concepts from CompTIA content guide question focus
- Maintains educational context

### 📚 **Comprehensive Coverage**
- 2,123 chunks of Google notes content
- Covers all IT Support topics
- Diverse question types and difficulty levels

### ⚡ **Efficiency**
- Optimized chunking (87 tokens average)
- Parallel processing capabilities
- Cached embeddings for fast retrieval

## Customization Options

### Question Types
- `multiple_choice`: 4-option questions
- `true_false`: True/false statements
- `short_answer`: Brief explanation questions

### Concept Extraction
- Adjust IT terminology dictionaries
- Modify entity recognition patterns
- Customize weighting schemes

### Matching Parameters
- Change similarity thresholds
- Adjust top-K retrieval count
- Modify embedding models

## Troubleshooting

### Common Issues

1. **Google Notes Not Found**
   ```bash
   python extract_google_notes_only.py
   ```

2. **API Key Issues**
   - Check `.env` file exists
   - Verify `GROQ_API_KEY` is set
   - Test with `python -c "from groq import Groq; print('OK')"`

3. **NLP Model Issues**
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. **Memory Issues**
   - Reduce `max_workers` parameter
   - Process smaller text chunks
   - Use smaller embedding models

## Performance Tips

### Optimization
- Use `max_workers=1` for testing
- Process text in smaller chunks
- Cache embeddings for repeated use

### Scaling
- Increase `max_workers` for parallel processing
- Use larger embedding models for better accuracy
- Implement batch processing for large datasets

## Integration Examples

### Web Application
```python
from flask import Flask, request, jsonify
from cross_reference_question_generator import CrossReferenceQuestionGenerator

app = Flask(__name__)
generator = CrossReferenceQuestionGenerator()

@app.route('/generate-questions', methods=['POST'])
def generate_questions():
    data = request.json
    results = generator.generate_cross_reference_questions(
        comptia_text=data['text'],
        num_questions=data.get('num_questions', 5)
    )
    return jsonify(results)
```

### Batch Processing
```python
import json
from pathlib import Path

# Process multiple CompTIA files
comptia_files = Path("comptia_content/").glob("*.txt")
for file_path in comptia_files:
    with open(file_path, 'r') as f:
        comptia_text = f.read()
    
    results = generator.generate_cross_reference_questions(comptia_text)
    
    # Save results
    output_file = f"questions_{file_path.stem}.json"
    generator.save_results(results, output_file)
```

## Future Enhancements

### Planned Features
- **Question Difficulty Assessment**: Automatic difficulty classification
- **Topic Clustering**: Group questions by specific topics
- **Answer Validation**: Cross-reference answers with multiple sources
- **Export Formats**: Support for CSV, Excel, and quiz platform formats

### Advanced Capabilities
- **Multi-language Support**: Process content in different languages
- **Custom Knowledge Bases**: Support for other training materials
- **Real-time Updates**: Dynamic content processing
- **Analytics Dashboard**: Question generation statistics and insights

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the demo scripts
3. Examine the test outputs
4. Verify your environment setup

The system is designed to be robust and user-friendly while maintaining high accuracy and relevance for IT certification training.
