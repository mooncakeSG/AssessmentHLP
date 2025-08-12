"""
Test script for Cross-Reference Question Generator
Demonstrates the system with various CompTIA content examples
"""

import os
import json
from cross_reference_question_generator import CrossReferenceQuestionGenerator

def test_with_sample_comptia_content():
    """Test with sample CompTIA content"""
    print("🧪 Testing Cross-Reference Question Generator")
    print("="*60)
    
    # Sample CompTIA content examples
    comptia_examples = {
        "network_security": """
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
        """,
        
        "hardware_troubleshooting": """
        Hardware Troubleshooting and Maintenance
        
        Computer hardware troubleshooting requires systematic approaches to 
        identify and resolve issues. Common hardware problems include RAM 
        failures, hard drive malfunctions, power supply issues, and CPU 
        overheating. Diagnostic tools such as POST (Power-On Self-Test) 
        provide initial hardware status information during system startup.
        
        Preventive maintenance includes regular cleaning of components, 
        monitoring system temperatures, and updating device drivers. Memory 
        testing utilities can identify RAM issues, while S.M.A.R.T. 
        (Self-Monitoring, Analysis, and Reporting Technology) monitors 
        hard drive health and predicts potential failures.
        
        When troubleshooting, technicians should follow the OSI model 
        approach: start with physical layer issues, then move to data 
        link, network, and application layers. Proper documentation of 
        troubleshooting steps and solutions is essential for future 
        reference and knowledge sharing.
        """,
        
        "operating_systems": """
        Operating System Fundamentals
        
        Operating systems serve as the interface between hardware and 
        software, managing system resources and providing services to 
        applications. Key OS functions include process management, memory 
        allocation, file system operations, and device driver management.
        
        Windows, Linux, and macOS are the primary operating systems used 
        in enterprise environments. Each has unique features: Windows 
        provides extensive hardware compatibility and user-friendly 
        interfaces, Linux offers open-source flexibility and server 
        optimization, while macOS combines Unix stability with Apple's 
        ecosystem integration.
        
        System administration tasks include user account management, 
        software installation and updates, backup procedures, and 
        performance monitoring. Command-line interfaces provide powerful 
        tools for system configuration and troubleshooting, while 
        graphical user interfaces offer accessibility for less technical users.
        """
    }
    
    try:
        # Initialize the generator
        print("🔄 Initializing Cross-Reference Question Generator...")
        generator = CrossReferenceQuestionGenerator()
        
        # Test each example
        for topic, content in comptia_examples.items():
            print(f"\n{'='*60}")
            print(f"📚 Testing: {topic.replace('_', ' ').title()}")
            print(f"{'='*60}")
            
            # Generate questions
            print("🔄 Generating questions...")
            results = generator.generate_cross_reference_questions(
                comptia_text=content,
                num_questions=3,
                question_types=['multiple_choice', 'true_false', 'short_answer']
            )
            
            # Display results
            if 'error' in results:
                print(f"❌ Error: {results['error']}")
                continue
                
            print(f"✅ Generation completed!")
            print(f"📊 Concepts extracted: {results['num_concepts_extracted']}")
            print(f"📄 Google notes chunks retrieved: {results['num_chunks_retrieved']}")
            print(f"❓ Questions generated: {len(results['questions'])}")
            
            # Show generated questions
            print(f"\n📋 Generated Questions:")
            for i, question in enumerate(results['questions'], 1):
                print(f"\nQ{i} ({question['type']}): {question['question']}")
                print(f"A{i}: {question['answer']}")
            
            # Save results for this topic
            output_file = f"cross_reference_{topic}_{int(time.time())}.json"
            generator.save_results(results, output_file)
            print(f"\n💾 Results saved to: {output_file}")
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

def test_concept_extraction():
    """Test concept extraction functionality"""
    print("\n🧪 Testing Concept Extraction")
    print("="*40)
    
    sample_text = """
    Cybersecurity Essentials
    
    Cybersecurity involves protecting systems, networks, and programs from 
    digital attacks. Common threats include malware, phishing, ransomware, 
    and social engineering attacks. Encryption algorithms like AES and RSA 
    provide data protection through mathematical transformations.
    
    Network security protocols such as SSL/TLS secure web communications, 
    while firewalls and intrusion detection systems monitor and control 
    network traffic. Authentication methods include passwords, biometrics, 
    and multi-factor authentication using tokens or mobile apps.
    """
    
    try:
        generator = CrossReferenceQuestionGenerator()
        
        # Extract concepts
        print("🔄 Extracting concepts...")
        concept_analysis = generator.extract_concepts_from_comptia(sample_text)
        
        print(f"✅ Extracted {len(concept_analysis['concepts'])} concepts")
        print(f"📊 Entities found: {len(concept_analysis['entities'])}")
        print(f"🔤 Noun phrases: {len(concept_analysis['noun_phrases'])}")
        print(f"💻 IT terms: {len(concept_analysis['it_terms'])}")
        
        # Show top concepts
        print(f"\n🏆 Top 10 Concepts:")
        for i, concept in enumerate(concept_analysis['concepts'][:10], 1):
            print(f"{i:2d}. {concept['text']} ({concept['type']}, weight: {concept['weight']})")
        
        # Show IT terms
        if concept_analysis['it_terms']:
            print(f"\n💻 IT Terms Found:")
            for term in concept_analysis['it_terms']:
                print(f"  - {term}")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def test_google_notes_retrieval():
    """Test Google notes retrieval functionality"""
    print("\n🧪 Testing Google Notes Retrieval")
    print("="*40)
    
    try:
        generator = CrossReferenceQuestionGenerator()
        
        # Test concepts
        test_concepts = [
            {'text': 'firewall', 'type': 'it_term', 'weight': 3},
            {'text': 'network security', 'type': 'noun_phrase', 'weight': 2},
            {'text': 'authentication', 'type': 'technical_term', 'weight': 2},
            {'text': 'encryption', 'type': 'it_term', 'weight': 3},
            {'text': 'malware', 'type': 'technical_term', 'weight': 2}
        ]
        
        print("🔄 Retrieving relevant Google notes...")
        relevant_chunks = generator.retrieve_relevant_google_notes(test_concepts, top_k=5)
        
        print(f"✅ Retrieved {len(relevant_chunks)} relevant chunks")
        
        # Show top chunks
        for i, chunk_data in enumerate(relevant_chunks, 1):
            chunk = chunk_data['chunk']
            similarity = chunk_data['similarity']
            print(f"\n📄 Chunk {i} (Similarity: {similarity:.3f}):")
            print(f"   Text: {chunk.get('text', '')[:200]}...")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main test function"""
    print("🚀 Cross-Reference Question Generator - Comprehensive Test")
    print("="*70)
    
    # Check if Google notes file exists
    google_notes_files = [
        "extracted_Google IT Support Professional Certificate Notes (1).json",
        "extracted_optimized_Google IT Support Professional Certificate Notes (1).json"
    ]
    
    found_file = None
    for file_path in google_notes_files:
        if os.path.exists(file_path):
            found_file = file_path
            break
    
    if not found_file:
        print("❌ Google notes file not found!")
        print("Please ensure you have extracted Google notes using the text_extractor.py")
        print("Available files:")
        for file in os.listdir('.'):
            if file.endswith('.json'):
                print(f"  - {file}")
        return
    
    print(f"✅ Found Google notes: {found_file}")
    
    # Run tests
    test_concept_extraction()
    test_google_notes_retrieval()
    test_with_sample_comptia_content()
    
    print(f"\n🎉 All tests completed!")

if __name__ == "__main__":
    import time
    main()
