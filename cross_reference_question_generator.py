"""
Cross-Reference Question Generator
Generates questions by extracting concepts from CompTIA content and matching with Google notes
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

class CrossReferenceQuestionGenerator:
    """
    Generates questions by cross-referencing CompTIA content with Google notes
    """
    
    def __init__(self, google_notes_path: str = None, max_workers: int = 4):
        """
        Initialize the cross-reference question generator
        
        Args:
            google_notes_path: Path to Google notes JSON file
            max_workers: Number of parallel workers for processing
        """
        self.max_workers = max_workers
        
        # Initialize NLP models
        logger.info("Loading NLP models...")
        try:
            self.nlp = spacy.load("en_core_web_sm")
            # Increase max length for large documents
            self.nlp.max_length = 2000000
        except OSError:
            logger.error("spaCy model not found. Please run: python -m spacy download en_core_web_sm")
            raise
        
        # Download NLTK data if not present
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger')
        try:
            nltk.data.find('chunkers/maxent_ne_chunker')
        except LookupError:
            nltk.download('maxent_ne_chunker')
        
        # Initialize embedding model
        logger.info("Loading embedding model...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize Groq client
        self.groq_client = Groq()
        
        # Load Google notes knowledge base
        self.google_notes = self._load_google_notes(google_notes_path)
        
        # IT-specific terminology for better concept extraction
        self.it_terminology = {
            'networking': ['router', 'switch', 'firewall', 'subnet', 'ip', 'dns', 'dhcp', 'vlan', 'wan', 'lan'],
            'hardware': ['cpu', 'ram', 'motherboard', 'gpu', 'ssd', 'hdd', 'psu', 'bios', 'uefi'],
            'operating_systems': ['windows', 'linux', 'macos', 'unix', 'kernel', 'shell', 'terminal'],
            'security': ['encryption', 'authentication', 'authorization', 'malware', 'virus', 'firewall', 'vpn'],
            'cloud': ['aws', 'azure', 'gcp', 'saas', 'paas', 'iaas', 'virtualization', 'container'],
            'databases': ['sql', 'mysql', 'postgresql', 'mongodb', 'database', 'query', 'index'],
            'programming': ['python', 'java', 'javascript', 'html', 'css', 'api', 'rest', 'json'],
            'certifications': ['comptia', 'a+', 'network+', 'security+', 'ccna', 'microsoft', 'cisco']
        }
        
        logger.info("Cross-reference question generator initialized successfully")
    
    def _load_google_notes(self, notes_path: str = None) -> Dict[str, Any]:
        """
        Load Google notes knowledge base from JSON file
        
        Args:
            notes_path: Path to Google notes JSON file
            
        Returns:
            Dictionary containing Google notes data
        """
        if notes_path is None:
            # Try to find Google notes file automatically
            possible_files = [
                "extracted_Google IT Support Professional Certificate Notes (1).json",
                "extracted_optimized_Google IT Support Professional Certificate Notes (1).json"
            ]
            
            for file_path in possible_files:
                if os.path.exists(file_path):
                    notes_path = file_path
                    break
        
        if notes_path is None or not os.path.exists(notes_path):
            logger.warning("Google notes file not found. Please provide a valid path.")
            return {'chunks': [], 'text': '', 'total_pages': 0}
        
        try:
            logger.info(f"Loading Google notes from: {notes_path}")
            with open(notes_path, 'r', encoding='utf-8') as f:
                notes_data = json.load(f)
            
            logger.info(f"Loaded Google notes: {len(notes_data.get('chunks', []))} chunks")
            return notes_data
            
        except Exception as e:
            logger.error(f"Error loading Google notes: {e}")
            return {'chunks': [], 'text': '', 'total_pages': 0}
    
    def extract_concepts_from_comptia(self, comptia_text: str) -> Dict[str, Any]:
        """
        Extract key concepts and keywords from CompTIA content using NLP
        
        Args:
            comptia_text: CompTIA textbook excerpt
            
        Returns:
            Dictionary containing extracted concepts, entities, and keywords
        """
        logger.info("Extracting concepts from CompTIA content...")
        
        # Clean and preprocess text
        cleaned_text = self._clean_text(comptia_text)
        
        # Process with spaCy
        doc = self.nlp(cleaned_text)
        
        # Extract named entities
        entities = []
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PRODUCT', 'GPE', 'PERSON', 'MISC']:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char
                })
        
        # Extract noun phrases
        noun_phrases = []
        for chunk in doc.noun_chunks:
            if len(chunk.text.strip()) > 2:  # Filter out very short phrases
                noun_phrases.append(chunk.text.strip())
        
        # Extract technical terms using POS tagging
        technical_terms = []
        for token in doc:
            if (token.pos_ in ['NOUN', 'PROPN'] and 
                len(token.text) > 2 and 
                not token.is_stop and
                token.text.lower() not in ['page', 'chapter', 'section']):
                technical_terms.append(token.text)
        
        # Extract IT-specific terminology
        it_terms = self._extract_it_terminology(cleaned_text)
        
        # Combine and rank concepts
        all_concepts = []
        
        # Add entities with high weight
        for entity in entities:
            all_concepts.append({
                'text': entity['text'],
                'type': 'entity',
                'weight': 3,
                'source': entity['label']
            })
        
        # Add IT terms with high weight
        for term in it_terms:
            all_concepts.append({
                'text': term,
                'type': 'it_term',
                'weight': 3,
                'source': 'it_terminology'
            })
        
        # Add noun phrases with medium weight
        for phrase in noun_phrases[:20]:  # Limit to top 20
            all_concepts.append({
                'text': phrase,
                'type': 'noun_phrase',
                'weight': 2,
                'source': 'noun_chunk'
            })
        
        # Add technical terms with medium weight
        for term in technical_terms[:30]:  # Limit to top 30
            all_concepts.append({
                'text': term,
                'type': 'technical_term',
                'weight': 2,
                'source': 'pos_tagging'
            })
        
        # Remove duplicates and sort by weight
        unique_concepts = {}
        for concept in all_concepts:
            key = concept['text'].lower().strip()
            if key not in unique_concepts or concept['weight'] > unique_concepts[key]['weight']:
                unique_concepts[key] = concept
        
        sorted_concepts = sorted(unique_concepts.values(), key=lambda x: x['weight'], reverse=True)
        
        logger.info(f"Extracted {len(sorted_concepts)} unique concepts from CompTIA content")
        
        return {
            'concepts': sorted_concepts[:50],  # Top 50 concepts
            'entities': entities,
            'noun_phrases': noun_phrases[:20],
            'technical_terms': technical_terms[:30],
            'it_terms': it_terms,
            'text_length': len(cleaned_text),
            'sentences': len(list(doc.sents))
        }
    
    def _extract_it_terminology(self, text: str) -> List[str]:
        """
        Extract IT-specific terminology from text
        
        Args:
            text: Input text
            
        Returns:
            List of IT terms found in text
        """
        text_lower = text.lower()
        found_terms = []
        
        for category, terms in self.it_terminology.items():
            for term in terms:
                if term.lower() in text_lower:
                    # Find the actual case in the original text
                    pattern = re.compile(re.escape(term), re.IGNORECASE)
                    matches = pattern.findall(text)
                    found_terms.extend(matches)
        
        return list(set(found_terms))  # Remove duplicates
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and preprocess text
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^(Page|Chapter|Section)\s+\d+', '', text, flags=re.MULTILINE)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        return text.strip()
    
    def retrieve_relevant_google_notes(self, concepts: List[Dict[str, Any]], 
                                     top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve relevant Google notes chunks based on extracted concepts
        
        Args:
            concepts: List of extracted concepts
            top_k: Number of top chunks to retrieve
            
        Returns:
            List of relevant Google notes chunks
        """
        logger.info(f"Retrieving relevant Google notes for {len(concepts)} concepts...")
        
        if not self.google_notes.get('chunks'):
            logger.warning("No Google notes chunks available")
            return []
        
        # Create query from concepts
        query_terms = [concept['text'] for concept in concepts[:20]]  # Use top 20 concepts
        query_text = ' '.join(query_terms)
        
        # Get embeddings for query
        query_embedding = self.embedder.encode([query_text])[0]
        
        # Calculate similarities with Google notes chunks
        similarities = []
        for chunk in self.google_notes['chunks']:
            if 'embedding' in chunk:
                chunk_embedding = np.array(chunk['embedding'])
                similarity = np.dot(query_embedding, chunk_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
                )
                similarities.append({
                    'chunk': chunk,
                    'similarity': similarity,
                    'score': similarity
                })
        
        # Sort by similarity and return top chunks
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        top_chunks = similarities[:top_k]
        
        logger.info(f"Retrieved {len(top_chunks)} relevant Google notes chunks")
        
        return top_chunks
    
    def generate_cross_reference_questions(self, comptia_text: str, 
                                         num_questions: int = 5,
                                         question_types: List[str] = None) -> Dict[str, Any]:
        """
        Generate questions by cross-referencing CompTIA content with Google notes
        
        Args:
            comptia_text: CompTIA textbook excerpt
            num_questions: Number of questions to generate
            question_types: Types of questions to generate
            
        Returns:
            Dictionary containing generated questions and metadata
        """
        logger.info("Starting cross-reference question generation...")
        
        if question_types is None:
            question_types = ['multiple_choice', 'true_false', 'short_answer']
        
        # Step 1: Extract concepts from CompTIA content
        concept_analysis = self.extract_concepts_from_comptia(comptia_text)
        
        # Step 2: Retrieve relevant Google notes
        relevant_chunks = self.retrieve_relevant_google_notes(concept_analysis['concepts'])
        
        if not relevant_chunks:
            logger.warning("No relevant Google notes found. Cannot generate questions.")
            return {
                'questions': [],
                'error': 'No relevant Google notes found',
                'concept_analysis': concept_analysis,
                'relevant_chunks': []
            }
        
        # Step 3: Generate questions using AI
        questions = self._generate_ai_questions(
            comptia_text, 
            relevant_chunks, 
            concept_analysis,
            num_questions,
            question_types
        )
        
        return {
            'questions': questions,
            'concept_analysis': concept_analysis,
            'relevant_chunks': relevant_chunks,
            'comptia_text_length': len(comptia_text),
            'num_concepts_extracted': len(concept_analysis['concepts']),
            'num_chunks_retrieved': len(relevant_chunks)
        }
    
    def _generate_ai_questions(self, comptia_text: str, 
                             relevant_chunks: List[Dict[str, Any]],
                             concept_analysis: Dict[str, Any],
                             num_questions: int,
                             question_types: List[str]) -> List[Dict[str, Any]]:
        """
        Generate questions using AI based on Google notes content
        
        Args:
            comptia_text: Original CompTIA text
            relevant_chunks: Retrieved Google notes chunks
            concept_analysis: Analysis of CompTIA concepts
            num_questions: Number of questions to generate
            question_types: Types of questions to generate
            
        Returns:
            List of generated questions
        """
        logger.info(f"Generating {num_questions} questions using AI...")
        
        # Prepare Google notes content
        google_content = self._prepare_google_content(relevant_chunks)
        
        # Prepare concepts for context
        top_concepts = [concept['text'] for concept in concept_analysis['concepts'][:15]]
        concepts_text = ', '.join(top_concepts)
        
        # Create AI prompt
        prompt = self._create_question_prompt(
            comptia_text, 
            google_content, 
            concepts_text,
            num_questions,
            question_types
        )
        
        try:
            # Generate questions using Groq
            completion = self.groq_client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2048,
                top_p=1,
                stream=False,
            )
            
            ai_response = completion.choices[0].message.content
            
            # Parse AI response
            questions = self._parse_ai_questions(ai_response, question_types)
            
            logger.info(f"Generated {len(questions)} questions successfully")
            return questions
            
        except Exception as e:
            logger.error(f"Error generating questions with AI: {e}")
            return [{
                'type': 'error',
                'question': f"Error generating questions: {str(e)}",
                'answer': '',
                'source': 'error'
            }]
    
    def _prepare_google_content(self, relevant_chunks: List[Dict[str, Any]]) -> str:
        """
        Prepare Google notes content for AI prompt
        
        Args:
            relevant_chunks: Retrieved Google notes chunks
            
        Returns:
            Formatted Google notes content
        """
        content_parts = []
        
        for i, chunk_data in enumerate(relevant_chunks[:5]):  # Use top 5 chunks
            chunk = chunk_data['chunk']
            content_parts.append(f"Google Notes Chunk {i+1} (Relevance: {chunk_data['similarity']:.3f}):")
            content_parts.append(chunk.get('text', '')[:500])  # Limit to 500 chars per chunk
            content_parts.append("")
        
        return "\n".join(content_parts)
    
    def _create_question_prompt(self, comptia_text: str, 
                              google_content: str,
                              concepts_text: str,
                              num_questions: int,
                              question_types: List[str]) -> str:
        """
        Create AI prompt for question generation
        
        Args:
            comptia_text: Original CompTIA text
            google_content: Google notes content
            concepts_text: Extracted concepts
            num_questions: Number of questions to generate
            question_types: Types of questions to generate
            
        Returns:
            Formatted AI prompt
        """
        type_instructions = {
            'multiple_choice': 'multiple choice questions with 4 options (A, B, C, D)',
            'true_false': 'true/false questions',
            'short_answer': 'short answer questions requiring brief explanations'
        }
        
        type_list = [type_instructions.get(qt, qt) for qt in question_types]
        type_text = ', '.join(type_list)
        
        prompt = f"""You are an expert IT certification instructor. Generate {num_questions} accurate quiz questions based STRICTLY on the Google notes content provided below.

IMPORTANT: All questions and answers must be based ONLY on the Google notes content, NOT on the CompTIA text. The CompTIA text is provided for context only.

CompTIA Context (for reference only):
{comptia_text[:1000]}...

Key Concepts Extracted: {concepts_text}

Google Notes Content (use this for questions):
{google_content}

Requirements:
1. Generate {num_questions} questions of the following types: {type_text}
2. Base ALL questions and answers STRICTLY on the Google notes content above
3. Do NOT use information from the CompTIA text for answers
4. Make questions practical and relevant to IT support
5. Ensure questions test understanding, not just memorization
6. Include clear, accurate answers

Format each question as:
Q1: [Question text]
A1: [Answer/explanation]

Q2: [Question text]
A2: [Answer/explanation]

... and so on.

Generate the questions now:"""

        return prompt
    
    def _parse_ai_questions(self, ai_response: str, question_types: List[str]) -> List[Dict[str, Any]]:
        """
        Parse AI response into structured questions
        
        Args:
            ai_response: Raw AI response
            question_types: Expected question types
            
        Returns:
            List of parsed questions
        """
        questions = []
        
        # Simple parsing based on Q/A format
        lines = ai_response.split('\n')
        current_question = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('Q') and ':' in line:
                # New question
                if current_question:
                    questions.append(current_question)
                
                question_text = line.split(':', 1)[1].strip()
                current_question = {
                    'type': self._determine_question_type(question_text, question_types),
                    'question': question_text,
                    'answer': '',
                    'source': 'google_notes'
                }
            
            elif line.startswith('A') and ':' in line and current_question:
                # Answer for current question
                answer_text = line.split(':', 1)[1].strip()
                current_question['answer'] = answer_text
        
        # Add the last question
        if current_question:
            questions.append(current_question)
        
        return questions
    
    def _determine_question_type(self, question_text: str, available_types: List[str]) -> str:
        """
        Determine the type of question based on content
        
        Args:
            question_text: Question text
            available_types: Available question types
            
        Returns:
            Question type
        """
        question_lower = question_text.lower()
        
        if 'multiple choice' in question_lower or any(opt in question_lower for opt in ['a)', 'b)', 'c)', 'd)']):
            return 'multiple_choice'
        elif 'true' in question_lower and 'false' in question_lower:
            return 'true_false'
        else:
            return 'short_answer'
    
    def save_results(self, results: Dict[str, Any], output_file: str = None) -> str:
        """
        Save generation results to file
        
        Args:
            results: Generation results
            output_file: Output file path
            
        Returns:
            Path to saved file
        """
        if output_file is None:
            timestamp = int(time.time())
            output_file = f"cross_reference_questions_{timestamp}.json"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Results saved to: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return None


def main():
    """
    Main function to demonstrate the cross-reference question generator
    """
    print("🚀 Cross-Reference Question Generator")
    print("="*60)
    
    # Sample CompTIA text (you can replace this with actual content)
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
    
    Security best practices include regular password updates, implementing 
    multi-factor authentication, and maintaining up-to-date security patches. 
    Network administrators must also monitor logs and implement proper access 
    controls to prevent unauthorized access to sensitive resources.
    """
    
    try:
        # Initialize the generator
        generator = CrossReferenceQuestionGenerator()
        
        # Generate questions
        print("🔄 Generating cross-reference questions...")
        results = generator.generate_cross_reference_questions(
            comptia_text=sample_comptia_text,
            num_questions=5,
            question_types=['multiple_choice', 'true_false', 'short_answer']
        )
        
        # Display results
        print(f"\n✅ Generation completed!")
        print(f"📊 Concepts extracted: {results['num_concepts_extracted']}")
        print(f"📄 Google notes chunks retrieved: {results['num_chunks_retrieved']}")
        print(f"❓ Questions generated: {len(results['questions'])}")
        
        # Show generated questions
        print(f"\n📋 Generated Questions:")
        for i, question in enumerate(results['questions'], 1):
            print(f"\nQ{i} ({question['type']}): {question['question']}")
            print(f"A{i}: {question['answer']}")
        
        # Save results
        output_file = generator.save_results(results)
        if output_file:
            print(f"\n💾 Results saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
