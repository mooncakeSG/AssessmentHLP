import os
import re
import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import cv2
import numpy as np
from pdfminer.high_level import extract_text_to_fp
from pdfminer.layout import LAParams
from io import StringIO
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer
from groq import Groq

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextExtractor:
    def __init__(self, tesseract_path: Optional[str] = None):
        # Tesseract setup
        self.tesseract_path = tesseract_path
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        self.ocr_config = '--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,;:!?()[]{}"\'-_+=<>/@#$%^&*|\\ '
        
        self.cleaning_patterns = {
            'page_numbers': r'^\s*\d+\s*$',
            'headers_footers': r'^(Page|Chapter|Section)\s+\d+',
            'excessive_whitespace': r'\s+',
            'orphaned_chars': r'^\s*[a-zA-Z]\s*$',
            'urls': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
        }
        
        # Initialize embedding model (Sentence-BERT)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize Groq client
        self.groq_client = Groq()

    def detect_pdf_type(self, pdf_path: str) -> str:
        try:
            doc = fitz.open(pdf_path)
            text_content = ""
            sample_pages = min(3, len(doc))
            for page_num in range(sample_pages):
                page = doc.load_page(page_num)
                text_content += page.get_text()
            doc.close()
            
            # Calculate ratio of text length to page content length (avoid div by zero)
            text_ratio = len(text_content.strip()) / max(1, len(text_content))
            
            if text_ratio > 0.1:
                return 'searchable'
            else:
                return 'scanned'
        except Exception as e:
            logger.error(f"Error detecting PDF type: {e}")
            return 'unknown'

    def extract_text_searchable(self, pdf_path: str) -> Dict[str, Any]:
        try:
            doc = fitz.open(pdf_path)
            fitz_text = ""
            page_texts = []
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                page_texts.append({
                    'page': page_num + 1,
                    'text': page_text,
                    'bbox': str(page.rect)
                })
                fitz_text += page_text + "\n"
            doc.close()

            output = StringIO()
            with open(pdf_path, 'rb') as pdf_file:
                extract_text_to_fp(pdf_file, output, laparams=LAParams())
            pdfminer_text = output.getvalue()
            output.close()

            combined_text = self._clean_text(fitz_text + "\n" + pdfminer_text)

            return {
                'text': combined_text,
                'pages': page_texts,
                'method': 'searchable',
                'total_pages': len(page_texts),
                'extraction_quality': 'high'
            }
        except Exception as e:
            logger.error(f"Error extracting searchable text: {e}")
            return {'text': '', 'error': str(e)}

    def ocr_page(self, page_num: int, doc: fitz.Document) -> Dict[str, Any]:
        page = doc.load_page(page_num)
        mat = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img_data = mat.tobytes("png")
        import io
        img = Image.open(io.BytesIO(img_data))
        img_processed = self._preprocess_image_for_ocr(img)
        page_text = pytesseract.image_to_string(img_processed, config=self.ocr_config)
        return {'page': page_num + 1, 'text': page_text, 'bbox': str(page.rect)}

    def extract_text_scanned_parallel(self, pdf_path: str) -> Dict[str, Any]:
        try:
            doc = fitz.open(pdf_path)
            page_texts = []
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(self.ocr_page, i, doc) for i in range(len(doc))]
                for future in futures:
                    page_texts.append(future.result())
            doc.close()

            full_text = "\n".join(p['text'] for p in sorted(page_texts, key=lambda x: x['page']))
            cleaned_text = self._clean_ocr_text(full_text)

            return {
                'text': cleaned_text,
                'pages': page_texts,
                'method': 'ocr_parallel',
                'total_pages': len(page_texts),
                'extraction_quality': 'medium'
            }
        except Exception as e:
            logger.error(f"Parallel OCR extraction error: {e}")
            return {'text': '', 'error': str(e)}

    def _preprocess_image_for_ocr(self, image: Image.Image) -> np.ndarray:
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        denoised = cv2.medianBlur(gray, 3)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((1,1), np.uint8)
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        return processed

    def _clean_text(self, text: str) -> str:
        text = re.sub(self.cleaning_patterns['page_numbers'], '', text, flags=re.MULTILINE)
        text = re.sub(self.cleaning_patterns['headers_footers'], '', text, flags=re.MULTILINE)
        text = re.sub(self.cleaning_patterns['urls'], '', text)
        text = re.sub(self.cleaning_patterns['excessive_whitespace'], ' ', text)
        lines = text.split('\n')
        cleaned_lines = [line for line in lines if not re.match(self.cleaning_patterns['orphaned_chars'], line.strip())]
        return '\n'.join(cleaned_lines).strip()

    def _clean_ocr_text(self, text: str) -> str:
        text = self._clean_text(text)
        text = re.sub(r'[|]{2,}', 'l', text)
        text = re.sub(r'[0O]{2,}', 'O', text)
        text = re.sub(r'[1l]{2,}', 'l', text)
        ocr_fixes = {'rn': 'm', 'cl': 'd', 'vv': 'w', '|': 'l'}
        for mistake, correction in ocr_fixes.items():
            text = text.replace(mistake, correction)
        return text

    def segment_content(self, text: str, max_chunk_size: int = 500) -> List[Dict[str, Any]]:
        """
        Segment content into smaller chunks optimized for API token limits
        Uses approximately 500 characters per chunk to stay well under 8K token limit
        """
        chunks = []
        paragraphs = text.split('\n\n')
        current_chunk = ""
        chunk_id = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            # If adding this paragraph would exceed the limit, save current chunk
            if len(current_chunk) + len(para) > max_chunk_size and current_chunk:
                chunks.append({
                    'id': chunk_id, 
                    'text': current_chunk.strip(), 
                    'length': len(current_chunk), 
                    'type': 'paragraph',
                    'estimated_tokens': self._estimate_tokens(current_chunk.strip())
                })
                chunk_id += 1
                current_chunk = para
            else:
                current_chunk += "\n\n" + para if current_chunk else para
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append({
                'id': chunk_id, 
                'text': current_chunk.strip(), 
                'length': len(current_chunk), 
                'type': 'paragraph',
                'estimated_tokens': self._estimate_tokens(current_chunk.strip())
            })
        
        # Further split chunks that are still too large
        final_chunks = []
        for chunk in chunks:
            if chunk['estimated_tokens'] > 6000:  # Conservative limit
                sub_chunks = self._split_large_chunk(chunk)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)
        
        return final_chunks
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Rough estimation of tokens (approximately 4 characters per token)
        This is a conservative estimate to stay well under API limits
        """
        return len(text) // 4
    
    def _split_large_chunk(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split a chunk that's still too large into smaller pieces
        """
        text = chunk['text']
        sentences = text.split('. ')
        sub_chunks = []
        current_text = ""
        sub_chunk_id = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Add period back if it was removed
            if not sentence.endswith('.'):
                sentence += '.'
                
            # If adding this sentence would exceed limit, save current sub-chunk
            if len(current_text) + len(sentence) > 400 and current_text:  # Smaller limit for sub-chunks
                sub_chunks.append({
                    'id': f"{chunk['id']}_{sub_chunk_id}",
                    'text': current_text.strip(),
                    'length': len(current_text),
                    'type': 'sentence_group',
                    'estimated_tokens': self._estimate_tokens(current_text.strip()),
                    'parent_chunk': chunk['id']
                })
                sub_chunk_id += 1
                current_text = sentence
            else:
                current_text += " " + sentence if current_text else sentence
        
        # Add the last sub-chunk if it exists
        if current_text:
            sub_chunks.append({
                'id': f"{chunk['id']}_{sub_chunk_id}",
                'text': current_text.strip(),
                'length': len(current_text),
                'type': 'sentence_group',
                'estimated_tokens': self._estimate_tokens(current_text.strip()),
                'parent_chunk': chunk['id']
            })
        
        return sub_chunks

    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        for chunk, emb in zip(chunks, embeddings):
            chunk['embedding'] = emb.tolist()
        return chunks

    def generate_questions(self, chunk_text: str) -> str:
        """
        Generate questions for a chunk with token limit validation
        """
        # Check if chunk is too large for API
        estimated_tokens = self._estimate_tokens(chunk_text)
        if estimated_tokens > 6000:  # Conservative limit
            logger.warning(f"Chunk too large ({estimated_tokens} estimated tokens), truncating...")
            # Truncate to safe size (approximately 4000 tokens)
            max_chars = 4000 * 4  # 4 chars per token estimate
            chunk_text = chunk_text[:max_chars] + "..."
            logger.info(f"Truncated chunk to {len(chunk_text)} characters")
        
        # Create a more focused prompt for smaller chunks
        prompt = f"""Generate 2-3 concise quiz questions based on this content:

{chunk_text}

Requirements:
- Create questions that test understanding of key concepts
- Include one multiple choice, one true/false, and one short answer question
- Make questions practical and relevant to IT certification
- Keep questions focused on the specific content provided

Format each question as:
Q1: [Question text]
A1: [Answer/explanation]

Q2: [Question text]  
A2: [Answer/explanation]

Q3: [Question text]
A3: [Answer/explanation]"""

        try:
            completion = self.groq_client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=512,  # Reduced for smaller chunks
                top_p=1,
                stream=False,
            )
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating questions: {e}")
            return f"Error generating questions: {str(e)}"

    def extract_text(self, pdf_path: str, save_chunks: bool = True) -> Dict[str, Any]:
        logger.info(f"Starting text extraction from: {pdf_path}")
        pdf_type = self.detect_pdf_type(pdf_path)
        logger.info(f"Detected PDF type: {pdf_type}")

        if pdf_type == 'searchable':
            extraction_result = self.extract_text_searchable(pdf_path)
        elif pdf_type == 'scanned':
            extraction_result = self.extract_text_scanned_parallel(pdf_path)
        else:
            extraction_result = self.extract_text_searchable(pdf_path)
            if not extraction_result.get('text'):
                extraction_result = self.extract_text_scanned_parallel(pdf_path)

        if 'error' in extraction_result:
            logger.error(f"Extraction failed: {extraction_result['error']}")
            return extraction_result

        chunks = self.segment_content(extraction_result['text'])
        logger.info(f"Created {len(chunks)} chunks from text")
        
        # Log chunk statistics
        total_tokens = sum(chunk.get('estimated_tokens', 0) for chunk in chunks)
        avg_tokens = total_tokens / len(chunks) if chunks else 0
        logger.info(f"Total estimated tokens: {total_tokens}, Average per chunk: {avg_tokens:.1f}")
        
        chunks = self.embed_chunks(chunks)
        extraction_result['chunks'] = chunks

        # Generate questions per chunk with progress tracking
        successful_questions = 0
        total_chunks = len(chunks)
        
        for i, chunk in enumerate(chunks):
            try:
                logger.info(f"Generating questions for chunk {i+1}/{total_chunks} (ID: {chunk['id']})")
                questions = self.generate_questions(chunk['text'])
                chunk['questions'] = questions
                successful_questions += 1
                logger.info(f"✅ Questions generated for chunk {chunk['id']}")
            except Exception as e:
                logger.error(f"Error generating questions for chunk {chunk['id']}: {e}")
                chunk['questions'] = ""
        
        logger.info(f"Question generation completed: {successful_questions}/{total_chunks} chunks successful")

        if save_chunks:
            self._save_extraction_results(pdf_path, extraction_result)

        logger.info(f"Extraction and question generation completed. {len(chunks)} chunks created.")
        return extraction_result

    def _save_extraction_results(self, pdf_path: str, results: Dict[str, Any]):
        try:
            pdf_name = Path(pdf_path).stem
            output_file = f"extracted_{pdf_name}.json"

            serializable_results = results.copy()
            if 'pages' in serializable_results:
                for page in serializable_results['pages']:
                    if 'bbox' in page:
                        page['bbox'] = str(page['bbox'])

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)

            logger.info(f"Extraction results saved to: {output_file}")

        except Exception as e:
            logger.error(f"Error saving extraction results: {e}")


def main():
    extractor = TextExtractor()
    pdf_files = [
        "Google IT Support Professional Certificate Notes (1).pdf",
        "CompTIA A+ Certification All-in-One Exam Guide , Tenth Edition (Exams 220-1001  220-1002) by Mike Meyers 3.pdf"
    ]

    for pdf_file in pdf_files:
        if os.path.exists(pdf_file):
            print(f"\n{'='*60}")
            print(f"Processing: {pdf_file}")
            print(f"{'='*60}")
            try:
                results = extractor.extract_text(pdf_file)
                if 'error' not in results:
                    print(f"✅ Extraction successful!")
                    print(f"📄 Total pages: {results.get('total_pages', 'Unknown')}")
                    print(f"📝 Text length: {len(results.get('text', ''))} characters")
                    print(f"🔧 Method: {results.get('method', 'Unknown')}")
                    print(f"📊 Quality: {results.get('extraction_quality', 'Unknown')}")
                    print(f"🧩 Chunks: {len(results.get('chunks', []))}")
                else:
                    print(f"❌ Extraction failed: {results['error']}")
            except Exception as e:
                print(f"❌ Error processing {pdf_file}: {e}")

if __name__ == "__main__":
    main()

