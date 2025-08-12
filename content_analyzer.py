"""
Content Analysis Module
Analyzes extracted text using NLP and embeddings to understand context and structure
"""

import os
import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path
import spacy
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContentAnalyzer:
    """
    Analyzes content using NLP and embeddings to extract key concepts and structure
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2') -> None:
        """
        Initialize the content analyzer

        Args:
            model_name: Sentence transformer model name
        """
        self.model_name = model_name

        # Load NLP models
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("✅ spaCy model loaded successfully")
        except OSError:
            logger.warning("⚠️ spaCy model not found. Installing...")
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        # Load sentence transformer
        try:
            self.sentence_model = SentenceTransformer(model_name)
            logger.info(f"✅ Sentence transformer model '{model_name}' loaded")
        except Exception as e:
            logger.error(f"❌ Error loading sentence transformer: {e}")
            raise

        # Download NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('averaged_perceptron_tagger')

        # Domain-specific terminology for IT
        self.it_terminology = {
            'networking': [
                'tcp', 'udp', 'ip', 'dns', 'dhcp', 'subnet', 'router', 'switch', 'firewall',
                'gateway', 'vlan', 'vpn', 'wifi', 'ethernet', 'bluetooth', 'nfc', 'cellular',
                'osi model', 'tcp/ip', 'ipv4', 'ipv6', 'mac address', 'port', 'protocol',
                'packet', 'frame', 'routing', 'switching', 'bandwidth', 'latency', 'throughput'
            ],
            'operating_systems': [
                'windows', 'linux', 'macos', 'kernel', 'process', 'thread', 'memory',
                'file system', 'registry', 'services', 'drivers', 'boot', 'partition',
                'virtual memory', 'swap', 'cache', 'buffer', 'interrupt', 'scheduler',
                'system call', 'api', 'shell', 'terminal', 'command line', 'gui'
            ],
            'security': [
                'authentication', 'authorization', 'encryption', 'malware', 'virus', 'firewall',
                'antivirus', 'spyware', 'ransomware', 'phishing', 'social engineering',
                'password', 'biometric', 'two-factor', 'ssl', 'tls', 'certificate',
                'vulnerability', 'patch', 'update', 'backup', 'recovery', 'incident response'
            ],
            'hardware': [
                'cpu', 'ram', 'gpu', 'motherboard', 'bios', 'uefi', 'storage',
                'ssd', 'hdd', 'optical drive', 'power supply', 'cooling', 'fan',
                'expansion card', 'pci', 'agp', 'usb', 'hdmi', 'displayport',
                'keyboard', 'mouse', 'printer', 'scanner', 'speaker', 'microphone'
            ],
            'troubleshooting': [
                'diagnostic', 'error', 'log', 'debug', 'repair', 'maintenance',
                'blue screen', 'crash', 'freeze', 'slow', 'performance', 'bottleneck',
                'conflict', 'compatibility', 'driver', 'firmware', 'update', 'rollback',
                'safe mode', 'recovery', 'restore', 'backup', 'data loss'
            ],
            'system_administration': [
                'user management', 'group policy', 'active directory', 'domain',
                'workgroup', 'permissions', 'access control', 'audit', 'monitoring',
                'performance', 'resource', 'capacity', 'scalability', 'availability',
                'redundancy', 'load balancing', 'clustering', 'virtualization'
            ]
        }

        # Question generation patterns
        self.question_patterns = {
            'definition': r'\b(what is|define|explain)\b',
            'process': r'\b(how to|steps|procedure)\b',
            'comparison': r'\b(difference|compare|versus)\b',
            'troubleshooting': r'\b(fix|resolve|error|problem)\b',
            'configuration': r'\b(setup|configure|install)\b'
        }

    def analyze_content(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main method to analyze extracted content

        Args:
            extracted_data: Output from text extractor

        Returns:
            Analysis results with concepts, structure, and metadata
        """
        logger.info("Starting content analysis...")

        text = extracted_data.get('text', '')
        chunks = extracted_data.get('chunks', [])

        if not text:
            logger.error("No text content to analyze")
            return {'error': 'No text content available'}

        # Check if embeddings are already provided by text extractor
        if chunks and 'embedding' in chunks[0]:
            logger.info("Using pre-generated embeddings from text extractor")
            embeddings = self._process_existing_embeddings(chunks)
        else:
            logger.info("Generating new embeddings")
            embeddings = self._generate_embeddings(chunks)
            
        concepts = self._extract_key_concepts(text)
        topics = self._identify_topics(chunks, embeddings)
        facts = self._extract_factual_statements(text)
        summary = self._generate_summary(text, concepts)

        analysis_results = {
            'concepts': concepts,
            'topics': topics,
            'facts': facts,
            'summary': summary,
            'embeddings': embeddings,
            'metadata': {
                'total_chunks': len(chunks),
                'total_concepts': len(concepts),
                'total_topics': len(topics),
                'total_facts': len(facts),
                'analysis_quality': 'high'
            }
        }

        logger.info(f"Content analysis completed. Found {len(concepts)} concepts, {len(topics)} topics, {len(facts)} facts.")

        return analysis_results

    def _generate_embeddings(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate embeddings for text chunks

        Args:
            chunks: List of text chunks

        Returns:
            Embeddings data
        """
        try:
            texts = [chunk['text'] for chunk in chunks]

            embeddings = self.sentence_model.encode(texts, show_progress_bar=True)
            similarities = cosine_similarity(embeddings)

            return {
                'embeddings': embeddings.tolist(),
                'similarities': similarities.tolist(),
                'chunk_ids': [chunk['id'] for chunk in chunks]
            }

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return {'error': str(e)}

    def _process_existing_embeddings(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process embeddings that were already generated by the text extractor

        Args:
            chunks: List of text chunks with pre-generated embeddings

        Returns:
            Processed embeddings data
        """
        try:
            embeddings = np.array([chunk['embedding'] for chunk in chunks])
            similarities = cosine_similarity(embeddings)

            return {
                'embeddings': embeddings.tolist(),
                'similarities': similarities.tolist(),
                'chunk_ids': [chunk['id'] for chunk in chunks],
                'source': 'text_extractor'
            }

        except Exception as e:
            logger.error(f"Error processing existing embeddings: {e}")
            return {'error': str(e)}

    def _extract_key_concepts(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract key concepts from text using NLP

        Args:
            text: Input text

        Returns:
            List of key concepts with metadata
        """
        try:
            doc = self.nlp(text)

            entities = {}
            for ent in doc.ents:
                if ent.label_ in ['ORG', 'PRODUCT', 'GPE', 'PERSON']:
                    entities.setdefault(ent.text, {'type': ent.label_, 'count': 0, 'sentences': []})
                    entities[ent.text]['count'] += 1

            noun_phrases = [chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text.split()) <= 4]

            phrase_freq = {}
            for phrase in noun_phrases:
                phrase_freq[phrase] = phrase_freq.get(phrase, 0) + 1

            tech_terms = []
            for domain, terms in self.it_terminology.items():
                for term in terms:
                    count = text.lower().count(term.lower())
                    if count:
                        tech_terms.append({'term': term, 'domain': domain, 'frequency': count})

            all_concepts = []

            for entity, data in entities.items():
                if data['count'] >= 2:
                    all_concepts.append({
                        'concept': entity,
                        'type': 'entity',
                        'category': data['type'],
                        'frequency': data['count'],
                        'importance': data['count'] * 2
                    })

            for phrase, freq in phrase_freq.items():
                if freq >= 3 and len(phrase) > 3:
                    all_concepts.append({
                        'concept': phrase,
                        'type': 'noun_phrase',
                        'category': 'general',
                        'frequency': freq,
                        'importance': freq
                    })

            for term_data in tech_terms:
                if term_data['frequency'] >= 2:
                    all_concepts.append({
                        'concept': term_data['term'],
                        'type': 'technical_term',
                        'category': term_data['domain'],
                        'frequency': term_data['frequency'],
                        'importance': term_data['frequency'] * 3
                    })

            all_concepts.sort(key=lambda x: x['importance'], reverse=True)

            return all_concepts[:50]

        except Exception as e:
            logger.error(f"Error extracting concepts: {e}")
            return []

    def _identify_topics(self, chunks: List[Dict[str, Any]], embeddings_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Identify topics by clustering similar chunks

        Args:
            chunks: Text chunks
            embeddings_data: Embeddings information

        Returns:
            List of identified topics
        """
        try:
            if 'error' in embeddings_data:
                return []

            embeddings = np.array(embeddings_data['embeddings'])
            
            # Adaptive clustering based on content diversity
            n_chunks = len(chunks)
            if n_chunks < 5:
                return []
            elif n_chunks < 10:
                n_clusters = 2
            elif n_chunks < 20:
                n_clusters = 3
            elif n_chunks < 50:
                n_clusters = min(8, n_chunks // 3)
            else:
                n_clusters = min(15, n_chunks // 4)

            # Use silhouette score to find optimal number of clusters
            from sklearn.metrics import silhouette_score
            best_score = -1
            best_n_clusters = n_clusters
            
            for k in range(2, min(n_clusters + 3, n_chunks)):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(embeddings)
                if len(set(cluster_labels)) > 1:  # At least 2 clusters
                    score = silhouette_score(embeddings, cluster_labels)
                    if score > best_score:
                        best_score = score
                        best_n_clusters = k

            logger.info(f"Using {best_n_clusters} clusters (silhouette score: {best_score:.3f})")
            
            kmeans = KMeans(n_clusters=best_n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)

            topics = []
            for cluster_id in range(best_n_clusters):
                cluster_chunks = [chunks[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
                if not cluster_chunks:
                    continue

                topic_text = ' '.join([chunk['text'] for chunk in cluster_chunks])
                topic_concepts = self._extract_key_concepts(topic_text)
                topic_name = self._generate_topic_name(topic_concepts, cluster_chunks)
                coherence = self._calculate_topic_coherence(cluster_chunks, embeddings_data)

                topics.append({
                    'id': cluster_id,
                    'name': topic_name,
                    'chunks': [chunk['id'] for chunk in cluster_chunks],
                    'concepts': topic_concepts[:10],
                    'size': len(cluster_chunks),
                    'coherence': coherence,
                    'silhouette_score': best_score
                })

            return topics

        except Exception as e:
            logger.error(f"Error identifying topics: {e}")
            return []

    def _extract_factual_statements(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract factual statements for question generation

        Args:
            text: Input text

        Returns:
            List of factual statements
        """
        try:
            sentences = nltk.sent_tokenize(text)
            facts = []

            for sentence in sentences:
                if self._is_factual_statement(sentence):
                    fact_type = self._classify_fact_type(sentence)
                    facts.append({
                        'statement': sentence,
                        'type': fact_type,
                        'confidence': self._calculate_fact_confidence(sentence),
                        'keywords': self._extract_fact_keywords(sentence)
                    })

            facts.sort(key=lambda x: x['confidence'], reverse=True)

            return facts[:100]

        except Exception as e:
            logger.error(f"Error extracting facts: {e}")
            return []

    def _generate_summary(self, text: str, concepts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate content summary

        Args:
            text: Input text
            concepts: Extracted concepts

        Returns:
            Summary information
        """
        if not text.strip():
            return {'key_concepts': [], 'summary_sentences': [], 'total_sentences': 0, 'main_topics': []}

        sentences = nltk.sent_tokenize(text)
        if not sentences:
            return {'key_concepts': [], 'summary_sentences': [], 'total_sentences': 0, 'main_topics': []}

        key_concepts = [c['concept'] for c in concepts[:10]]

        if len(sentences) > 10:
            sentence_embeddings = self.sentence_model.encode(sentences)
            doc_embedding = self.sentence_model.encode([text])
            similarities = cosine_similarity([doc_embedding[0]], sentence_embeddings)[0]
            top_indices = similarities.argsort()[-5:]
            summary_sentences = [sentences[i] for i in sorted(top_indices)]
        else:
            summary_sentences = sentences[:5]

        return {
            'key_concepts': key_concepts,
            'summary_sentences': summary_sentences,
            'total_sentences': len(sentences),
            'main_topics': [c['category'] for c in concepts[:5]]
        }

    def _is_factual_statement(self, sentence: str) -> bool:
        """
        Check if a sentence contains factual information

        Args:
            sentence: Input sentence

        Returns:
            True if factual
        """
        factual_indicators = [
            r'\bis\b', r'\bare\b', r'\bwas\b', r'\bwere\b', r'\bhas\b', r'\bhave\b', r'\bhad\b',
            'consists of', 'comprises', 'includes', 'contains',
            'defined as', 'refers to', 'means', 'represents',
            r'\b[A-Z]{2,}\b means',  # Acronym definitions
        ]

        sentence_lower = sentence.lower()
        for pattern in factual_indicators:
            if re.search(pattern, sentence_lower):
                return True
        return False

    def _classify_fact_type(self, sentence: str) -> str:
        """
        Classify the type of factual statement

        Args:
            sentence: Input sentence

        Returns:
            Fact type
        """
        sentence_lower = sentence.lower()
        if any(word in sentence_lower for word in ['definition', 'defined', 'means', 'refers']):
            return 'definition'
        elif any(word in sentence_lower for word in ['process', 'procedure', 'steps', 'method']):
            return 'process'
        elif any(word in sentence_lower for word in ['difference', 'compare', 'versus', 'unlike']):
            return 'comparison'
        elif any(word in sentence_lower for word in ['error', 'problem', 'issue', 'troubleshoot']):
            return 'troubleshooting'
        elif any(word in sentence_lower for word in ['configure', 'setup', 'install']):
            return 'configuration'
        else:
            return 'general'

    def _calculate_fact_confidence(self, sentence: str) -> float:
        """
        Calculate confidence score for a factual statement

        Args:
            sentence: Input sentence

        Returns:
            Confidence score (0-1)
        """
        confidence = 0.5
        words = sentence.split()

        if 10 <= len(words) <= 30:
            confidence += 0.2

        tech_terms = sum(
            1 for domain in self.it_terminology.values()
            for term in domain if term.lower() in sentence.lower()
        )
        confidence += min(0.3, tech_terms * 0.1)

        if self._is_factual_statement(sentence):
            confidence += 0.2

        return min(1.0, confidence)

    def _extract_fact_keywords(self, sentence: str) -> List[str]:
        """
        Extract keywords from a factual statement

        Args:
            sentence: Input sentence

        Returns:
            List of keywords
        """
        doc = self.nlp(sentence)
        keywords = [token.text.lower() for token in doc if token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and not token.is_stop]
        return keywords[:5]

    def _generate_topic_name(self, concepts: List[Dict[str, Any]], chunks: List[Dict[str, Any]]) -> str:
        """
        Generate a name for a topic based on its concepts

        Args:
            concepts: Topic concepts
            chunks: Topic chunks

        Returns:
            Topic name
        """
        if not concepts:
            return f"Topic_{len(chunks)}"

        # Look for technical terms first
        tech_concepts = [c for c in concepts if c['type'] == 'technical_term']
        if tech_concepts:
            # Combine top technical terms
            top_tech = tech_concepts[:2]
            if len(top_tech) == 2:
                return f"{top_tech[0]['concept'].title()} and {top_tech[1]['concept'].title()}"
            else:
                return top_tech[0]['concept'].title()

        # Look for entities
        entities = [c for c in concepts if c['type'] == 'entity']
        if entities:
            return entities[0]['concept'].title()

        # Look for noun phrases
        noun_phrases = [c for c in concepts if c['type'] == 'noun_phrase']
        if noun_phrases:
            # Clean up the noun phrase
            phrase = noun_phrases[0]['concept']
            # Remove common stop words and clean up
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            words = [word for word in phrase.split() if word.lower() not in stop_words]
            if words:
                return ' '.join(words).title()

        # Fallback to first concept
        top_concept = concepts[0]['concept']
        name = top_concept.replace('_', ' ').title()
        return name

    def _calculate_topic_coherence(self, chunks: List[Dict[str, Any]], embeddings_data: Dict[str, Any]) -> float:
        """
        Calculate topic coherence score

        Args:
            chunks: Topic chunks
            embeddings_data: Embeddings information

        Returns:
            Coherence score (0-1)
        """
        try:
            if 'error' in embeddings_data or len(chunks) < 2:
                return 0.5

            chunk_ids = [chunk['id'] for chunk in chunks]
            chunk_indices = [embeddings_data['chunk_ids'].index(cid) for cid in chunk_ids]

            embeddings = np.array(embeddings_data['embeddings'])
            topic_embeddings = embeddings[chunk_indices]

            similarities = cosine_similarity(topic_embeddings)
            coherence = np.mean(similarities[np.triu_indices_from(similarities, k=1)])

            return float(coherence)

        except Exception as e:
            logger.error(f"Error calculating topic coherence: {e}")
            return 0.5

    def save_analysis_results(self, results: Dict[str, Any], output_file: str) -> None:
        """
        Save analysis results to JSON file

        Args:
            results: Analysis results
            output_file: Output file path
        """
        try:
            serializable_results = results.copy()
            
            # Optimize embeddings storage for large datasets
            if 'embeddings' in serializable_results:
                embeddings_data = serializable_results['embeddings']
                if 'embeddings' in embeddings_data:
                    # Store only metadata for large embeddings
                    embeddings_array = np.array(embeddings_data['embeddings'])
                    serializable_results['embeddings'] = {
                        'shape': embeddings_array.shape,
                        'chunk_ids': embeddings_data['chunk_ids'],
                        'source': embeddings_data.get('source', 'generated'),
                        'total_size_mb': embeddings_array.nbytes / (1024 * 1024)
                    }
                    
                    # Save embeddings separately if they're large
                    if embeddings_array.nbytes > 10 * 1024 * 1024:  # > 10MB
                        embeddings_file = output_file.replace('.json', '_embeddings.npy')
                        np.save(embeddings_file, embeddings_array)
                        serializable_results['embeddings']['separate_file'] = embeddings_file
                        logger.info(f"Large embeddings saved separately to: {embeddings_file}")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            logger.info(f"Analysis results saved to: {output_file}")
        except Exception as e:
            logger.error(f"Error saving analysis results: {e}")

    def load_embeddings_from_file(self, embeddings_file: str) -> np.ndarray:
        """
        Load embeddings from separate file

        Args:
            embeddings_file: Path to embeddings file

        Returns:
            Loaded embeddings array
        """
        try:
            embeddings = np.load(embeddings_file)
            logger.info(f"Loaded embeddings from: {embeddings_file}")
            return embeddings
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            return np.array([])


def main() -> None:
    """Test the content analyzer"""
    analyzer = ContentAnalyzer()

    sample_data = {
        'text': """
        TCP (Transmission Control Protocol) is a connection-oriented protocol that provides reliable data transmission.
        It establishes a connection before sending data and ensures that all packets arrive in the correct order.
        UDP (User Datagram Protocol) is a connectionless protocol that does not guarantee delivery or order.
        Firewalls are network security devices that monitor and control incoming and outgoing network traffic.
        """,
        'chunks': [
            {'id': 0, 'text': 'TCP (Transmission Control Protocol) is a connection-oriented protocol that provides reliable data transmission.', 'length': 100},
            {'id': 1, 'text': 'It establishes a connection before sending data and ensures that all packets arrive in the correct order.', 'length': 120},
            {'id': 2, 'text': 'UDP (User Datagram Protocol) is a connectionless protocol that does not guarantee delivery or order.', 'length': 110},
            {'id': 3, 'text': 'Firewalls are network security devices that monitor and control incoming and outgoing network traffic.', 'length': 130}
        ]
    }

    print("Testing Content Analyzer...")
    try:
        results = analyzer.analyze_content(sample_data)
        if 'error' not in results:
            print("✅ Analysis completed successfully!")
            print(f"📊 Found {len(results['concepts'])} concepts")
            print(f"📚 Identified {len(results['topics'])} topics")
            print(f"📝 Extracted {len(results['facts'])} facts")
            print(f"📋 Summary generated")
        else:
            print(f"❌ Analysis failed: {results['error']}")
    except Exception as e:
        logger.error(f"Unexpected error during analysis: {e}")


if __name__ == "__main__":
    main()
