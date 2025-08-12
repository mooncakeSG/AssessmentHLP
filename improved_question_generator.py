"""
Improved Cross-Reference Question Generator
Fixes quality issues and ensures all questions have complete answers
"""

import os
import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import re
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import time

# NLP Libraries
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk

# AI and Embeddings
from sentence_transformers import SentenceTransformer
from groq import Groq

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImprovedCrossReferenceQuestionGenerator:
    """
    Improved Cross-Reference Question Generator with quality fixes
    """
    
    def __init__(self, google_notes_path: str = None, max_workers: int = 4):
        """
        Initialize the improved question generator
        
        Args:
            google_notes_path: Path to Google notes JSON file
            max_workers: Number of parallel workers
        """
        self.max_workers = max_workers
        
        # Initialize NLP models
        logger.info("Loading NLP models...")
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Installing...")
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        
        # Initialize embedding model
        logger.info("Loading embedding model...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize Groq client
        self.groq_client = Groq()
        
        # Load Google notes
        if google_notes_path:
            self.load_google_notes(google_notes_path)
        else:
            self.google_notes = None
        
        # IT terminology for concept extraction
        self.it_terminology = {
            'networking': ['router', 'switch', 'firewall', 'vpn', 'dns', 'dhcp', 'tcp', 'ip', 'subnet', 'gateway'],
            'security': ['encryption', 'authentication', 'authorization', 'ssl', 'tls', 'ids', 'ips', 'malware', 'virus'],
            'hardware': ['cpu', 'ram', 'motherboard', 'gpu', 'ssd', 'hdd', 'power supply', 'cooling'],
            'software': ['os', 'application', 'driver', 'firmware', 'bios', 'uefi', 'kernel'],
            'troubleshooting': ['diagnostic', 'post', 'error code', 'blue screen', 'freeze', 'crash']
        }
        
        logger.info("Improved Cross-reference question generator initialized successfully")
    
    def load_google_notes(self, file_path: str):
        """Load Google notes from JSON file"""
        logger.info(f"Loading Google notes from: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.google_notes = data.get('chunks', [])
            logger.info(f"Loaded Google notes: {len(self.google_notes)} chunks")
            
        except Exception as e:
            logger.error(f"Error loading Google notes: {e}")
            self.google_notes = []
    
    def generate_cross_reference_questions(self, comptia_text: str, 
                                         num_questions: int = 5,
                                         question_types: List[str] = None) -> Dict[str, Any]:
        """
        Generate questions using improved quality control
        
        Args:
            comptia_text: CompTIA textbook content
            num_questions: Number of questions to generate
            question_types: Types of questions to generate
            
        Returns:
            Dictionary with generated questions and metadata
        """
        logger.info("Starting improved cross-reference question generation...")
        
        if not self.google_notes:
            return {'error': 'Google notes not loaded'}
        
        # Extract concepts from CompTIA text
        logger.info("Extracting concepts from CompTIA content...")
        concept_analysis = self._extract_concepts(comptia_text)
        
        # Retrieve relevant Google notes chunks
        logger.info(f"Retrieving relevant Google notes for {len(concept_analysis['concepts'])} concepts...")
        relevant_chunks = self._retrieve_relevant_chunks(concept_analysis['concepts'])
        
        # Generate questions with improved quality control
        logger.info(f"Generating {num_questions} questions using improved AI...")
        questions = self._generate_questions_improved(
            comptia_text, relevant_chunks, concept_analysis, num_questions, question_types
        )
        
        # Validate and fix any questions without answers
        questions = self._validate_and_fix_questions(questions)
        
        return {
            'questions': questions,
            'concept_analysis': concept_analysis,
            'relevant_chunks': relevant_chunks,
            'comptia_text_length': len(comptia_text),
            'num_concepts_extracted': len(concept_analysis['concepts']),
            'num_chunks_retrieved': len(relevant_chunks)
        }
    
    def _extract_concepts(self, text: str) -> Dict[str, Any]:
        """Extract concepts using improved NLP techniques"""
        # Process text with spaCy
        doc = self.nlp(text)
        
        concepts = []
        entities = []
        noun_phrases = []
        technical_terms = []
        it_terms = []
        
        # Extract named entities
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'type': ent.label_,
                'weight': 3
            })
        
        # Extract noun phrases
        for chunk in doc.noun_chunks:
            noun_phrases.append({
                'text': chunk.text,
                'type': 'noun_phrase',
                'weight': 2
            })
        
        # Extract technical terms
        text_lower = text.lower()
        for category, terms in self.it_terminology.items():
            for term in terms:
                if term.lower() in text_lower:
                    technical_terms.append({
                        'text': term,
                        'type': 'technical_term',
                        'weight': 3
                    })
        
        # Combine all concepts
        all_concepts = entities + noun_phrases + technical_terms + it_terms
        
        # Remove duplicates and sort by weight
        unique_concepts = []
        seen = set()
        for concept in all_concepts:
            if concept['text'].lower() not in seen:
                unique_concepts.append(concept)
                seen.add(concept['text'].lower())
        
        # Sort by weight and take top concepts
        unique_concepts.sort(key=lambda x: x['weight'], reverse=True)
        
        return {
            'concepts': unique_concepts[:50],  # Top 50 concepts
            'entities': entities,
            'noun_phrases': noun_phrases,
            'technical_terms': technical_terms,
            'it_terms': it_terms
        }
    
    def _retrieve_relevant_chunks(self, concepts: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve relevant Google notes chunks using semantic similarity"""
        if not concepts:
            return []
        
        # Get concept texts
        concept_texts = [concept['text'] for concept in concepts]
        
        # Generate embeddings for concepts
        concept_embeddings = self.embedder.encode(concept_texts, convert_to_numpy=True)
        
        # Get embeddings for Google notes chunks
        chunk_texts = [chunk.get('text', '') for chunk in self.google_notes]
        chunk_embeddings = self.embedder.encode(chunk_texts, convert_to_numpy=True)
        
        # Calculate similarities
        similarities = np.dot(chunk_embeddings, concept_embeddings.T)
        
        # Get top chunks for each concept
        top_chunks = []
        for i in range(len(concepts)):
            top_indices = np.argsort(similarities[:, i])[-top_k:][::-1]
            for idx in top_indices:
                similarity = similarities[idx, i]
                if similarity > 0.3:  # Minimum similarity threshold
                    top_chunks.append({
                        'chunk': self.google_notes[idx],
                        'similarity': float(similarity),
                        'score': float(similarity)
                    })
        
        # Remove duplicates and sort by similarity
        unique_chunks = []
        seen = set()
        for chunk_data in top_chunks:
            chunk_id = chunk_data['chunk'].get('id', '')
            if chunk_id not in seen:
                unique_chunks.append(chunk_data)
                seen.add(chunk_id)
        
        unique_chunks.sort(key=lambda x: x['similarity'], reverse=True)
        return unique_chunks[:top_k]
    
    def _generate_questions_improved(self, comptia_text: str, 
                                   relevant_chunks: List[Dict[str, Any]],
                                   concept_analysis: Dict[str, Any],
                                   num_questions: int,
                                   question_types: List[str]) -> List[Dict[str, Any]]:
        """Generate questions with improved quality control"""
        logger.info(f"Generating {num_questions} questions using improved AI...")
        
        # Prepare Google notes content
        google_content = self._prepare_google_content_improved(relevant_chunks)
        
        # Prepare concepts for context
        top_concepts = [concept['text'] for concept in concept_analysis['concepts'][:15]]
        concepts_text = ', '.join(top_concepts)
        
        # Create improved AI prompt
        prompt = self._create_improved_prompt(
            comptia_text, 
            google_content, 
            concepts_text,
            num_questions,
            question_types
        )
        
        try:
            # Generate questions using Groq with retry logic
            questions = []
            max_retries = 3
            
            for attempt in range(max_retries):
                try:
                    completion = self.groq_client.chat.completions.create(
                        model="openai/gpt-oss-20b",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.7,
                        max_tokens=2048,
                        top_p=1,
                        stream=False,
                    )
                    
                    ai_response = completion.choices[0].message.content
                    questions = self._parse_ai_questions_improved(ai_response, question_types)
                    
                    # Validate that we got the expected number of questions
                    if len(questions) >= num_questions:
                        break
                    else:
                        logger.warning(f"Attempt {attempt + 1}: Got {len(questions)} questions, expected {num_questions}")
                        
                except Exception as e:
                    logger.error(f"Attempt {attempt + 1} failed: {e}")
                    if attempt == max_retries - 1:
                        raise
            
            logger.info(f"Generated {len(questions)} questions successfully")
            return questions[:num_questions]  # Return only requested number
            
        except Exception as e:
            logger.error(f"Error generating questions with AI: {e}")
            return [{
                'type': 'error',
                'question': f"Error generating questions: {str(e)}",
                'answer': 'Please try again with different content.',
                'source': 'error'
            }]
    
    def _prepare_google_content_improved(self, relevant_chunks: List[Dict[str, Any]]) -> str:
        """Prepare Google notes content with better formatting"""
        content_parts = []
        
        for i, chunk_data in enumerate(relevant_chunks[:5]):  # Use top 5 chunks
            chunk = chunk_data['chunk']
            content_parts.append(f"=== Google Notes Chunk {i+1} (Relevance: {chunk_data['similarity']:.3f}) ===")
            content_parts.append(chunk.get('text', '')[:600])  # Limit to 600 chars per chunk
            content_parts.append("")
        
        return "\n".join(content_parts)
    
    def _create_improved_prompt(self, comptia_text: str, 
                              google_content: str,
                              concepts_text: str,
                              num_questions: int,
                              question_types: List[str]) -> str:
        """Create improved AI prompt for better question generation"""
        type_instructions = {
            'multiple_choice': 'multiple choice questions with 4 options (A, B, C, D) and clear correct answer',
            'true_false': 'true/false questions with clear true or false answer',
            'short_answer': 'short answer questions requiring brief but complete explanations'
        }
        
        type_list = [type_instructions.get(qt, qt) for qt in question_types]
        type_text = ', '.join(type_list)
        
        prompt = f"""You are an expert IT certification instructor. Generate exactly {num_questions} high-quality quiz questions based STRICTLY on the Google notes content provided below.

CRITICAL REQUIREMENTS:
1. ALL questions and answers must be based ONLY on the Google notes content
2. EVERY question MUST have a complete, accurate answer
3. Do NOT use information from the CompTIA text for answers
4. Make questions practical and relevant to IT support
5. Ensure questions test understanding, not just memorization

CompTIA Context (for reference only):
{comptia_text[:800]}...

Key Concepts Extracted: {concepts_text}

Google Notes Content (use this for questions):
{google_content}

Generate exactly {num_questions} questions of the following types: {type_text}

IMPORTANT FORMAT REQUIREMENTS:
- Each question MUST be followed by its answer
- Use this exact format for each question:

Q1: [Question text]
A1: [Complete answer/explanation]

Q2: [Question text]
A2: [Complete answer/explanation]

... and so on for all {num_questions} questions.

QUALITY REQUIREMENTS:
- Every question must have a complete answer
- Answers should be 1-3 sentences for short answer questions
- Multiple choice questions must have 4 options and clear correct answer
- True/false questions must have clear true or false answer
- Base all content on the Google notes provided

Generate the questions now:"""

        return prompt
    
    def _parse_ai_questions_improved(self, ai_response: str, question_types: List[str]) -> List[Dict[str, Any]]:
        """Parse AI response with improved error handling"""
        questions = []
        
        # Enhanced parsing with multiple patterns
        patterns = [
            r'Q(\d+):\s*(.+?)(?=Q\d+:|A\d+:|$)',  # Q1: question
            r'A(\d+):\s*(.+?)(?=Q\d+:|A\d+:|$)',  # A1: answer
        ]
        
        # Extract questions and answers
        question_matches = re.findall(r'Q(\d+):\s*(.+?)(?=Q\d+:|A\d+:|$)', ai_response, re.DOTALL)
        answer_matches = re.findall(r'A(\d+):\s*(.+?)(?=Q\d+:|A\d+:|$)', ai_response, re.DOTALL)
        
        # Create mapping of question numbers to answers
        answer_map = {num: answer.strip() for num, answer in answer_matches}
        
        # Process questions
        for num, question_text in question_matches:
            question_text = question_text.strip()
            answer_text = answer_map.get(num, '').strip()
            
            # Ensure we have both question and answer
            if question_text and answer_text:
                questions.append({
                    'type': self._determine_question_type(question_text, question_types),
                    'question': question_text,
                    'answer': answer_text,
                    'source': 'google_notes'
                })
        
        return questions
    
    def _validate_and_fix_questions(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and fix questions without answers"""
        fixed_questions = []
        
        for i, question in enumerate(questions):
            if not question.get('answer', '').strip():
                logger.warning(f"Question {i+1} has no answer, attempting to fix...")
                
                # Try to generate a simple answer based on the question
                fixed_answer = self._generate_simple_answer(question['question'])
                question['answer'] = fixed_answer
                
                logger.info(f"Generated answer for question {i+1}: {fixed_answer[:50]}...")
            
            fixed_questions.append(question)
        
        return fixed_questions
    
    def _generate_simple_answer(self, question_text: str) -> str:
        """Generate a simple answer for questions without answers"""
        try:
            # Create a simple prompt to generate an answer
            prompt = f"""Based on the question below, provide a brief, accurate answer:

Question: {question_text}

Answer:"""
            
            completion = self.groq_client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=100,
                top_p=1,
                stream=False,
            )
            
            return completion.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating simple answer: {e}")
            return "Answer not available. Please refer to the Google notes for this information."
    
    def _determine_question_type(self, question_text: str, available_types: List[str]) -> str:
        """Determine question type with improved logic"""
        question_lower = question_text.lower()
        
        if 'multiple choice' in question_lower or any(opt in question_lower for opt in ['a)', 'b)', 'c)', 'd)']):
            return 'multiple_choice'
        elif ('true' in question_lower and 'false' in question_lower) or question_lower.startswith('true or false'):
            return 'true_false'
        else:
            return 'short_answer'
    
    def save_results(self, results: Dict[str, Any], output_file: str = None) -> str:
        """Save results to file"""
        if output_file is None:
            timestamp = int(time.time())
            output_file = f"improved_questions_{timestamp}.json"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Results saved to: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return None

def main():
    """Test the improved question generator"""
    print("🚀 Testing Improved Cross-Reference Question Generator")
    print("="*60)
    
    # Initialize generator
    generator = ImprovedCrossReferenceQuestionGenerator()
    
    # Load Google notes
    google_notes_file = "extracted_Google IT Support Professional Certificate Notes (1).json"
    if os.path.exists(google_notes_file):
        generator.load_google_notes(google_notes_file)
    else:
        print(f"❌ Google notes file not found: {google_notes_file}")
        return
    
    # Sample CompTIA content
    sample_comptia_text = """
    Network Security Fundamentals
    
    Network security is a critical component of modern IT infrastructure. 
    Firewalls act as the first line of defense, filtering traffic based on 
    predefined rules. VPNs (Virtual Private Networks) provide secure remote 
    access by encrypting data transmission. Intrusion Detection Systems (IDS) 
    monitor network traffic for suspicious activity and potential threats.
    
    Common network protocols include TCP/IP, which provides reliable data 
    transmission, and DNS (Domain Name System), which translates domain names 
    to IP addresses. DHCP (Dynamic Host Configuration Protocol) automatically 
    assigns IP addresses to devices on the network.
    """
    
    # Generate questions
    print("🔄 Generating questions with improved quality control...")
    results = generator.generate_cross_reference_questions(
        comptia_text=sample_comptia_text,
        num_questions=5,
        question_types=['multiple_choice', 'true_false', 'short_answer']
    )
    
    # Display results
    print(f"\n✅ Generated {len(results['questions'])} questions")
    
    for i, question in enumerate(results['questions'], 1):
        print(f"\nQ{i} ({question['type'].replace('_', ' ').title()}):")
        print(f"   {question['question']}")
        print(f"   Answer: {question['answer']}")
    
    # Save results
    output_file = generator.save_results(results)
    if output_file:
        print(f"\n💾 Results saved to: {output_file}")

if __name__ == "__main__":
    main()
