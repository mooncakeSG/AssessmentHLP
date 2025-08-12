"""
Cross-Reference Question Generator Demo
Demonstrates how to use the system with CompTIA content and Google notes
"""

import os
import json
from cross_reference_question_generator import CrossReferenceQuestionGenerator

def demo_cross_reference_system():
    """Demonstrate the cross-reference question generation system"""
    print("🚀 Cross-Reference Question Generator Demo")
    print("="*60)
    print("This demo shows how to generate questions by:")
    print("1. Extracting concepts from CompTIA content")
    print("2. Matching with Google IT Support notes")
    print("3. Generating questions based on Google notes content")
    print("="*60)
    
    # Check if Google notes file exists
    google_notes_file = "extracted_Google IT Support Professional Certificate Notes (1).json"
    if not os.path.exists(google_notes_file):
        print(f"❌ Google notes file not found: {google_notes_file}")
        print("Please run extract_google_notes_only.py first")
        return
    
    print(f"✅ Found Google notes: {google_notes_file}")
    
    # Initialize the generator
    print("\n🔄 Initializing Cross-Reference Question Generator...")
    generator = CrossReferenceQuestionGenerator()
    
    # Sample CompTIA content (you can replace this with actual CompTIA text)
    comptia_content = """
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
    
    print(f"\n📚 CompTIA Content Sample:")
    print(f"Length: {len(comptia_content)} characters")
    print(f"Topics: Network Security, Firewalls, VPNs, IDS, TCP/IP, DNS, DHCP")
    
    # Generate questions
    print(f"\n🔄 Generating cross-reference questions...")
    results = generator.generate_cross_reference_questions(
        comptia_text=comptia_content,
        num_questions=5,
        question_types=['multiple_choice', 'true_false', 'short_answer']
    )
    
    # Display results
    print(f"\n✅ Generation completed!")
    print(f"📊 Concepts extracted: {results['num_concepts_extracted']}")
    print(f"📄 Google notes chunks retrieved: {results['num_chunks_retrieved']}")
    print(f"❓ Questions generated: {len(results['questions'])}")
    
    # Show generated questions
    print(f"\n📋 Generated Questions (based on Google notes):")
    for i, question in enumerate(results['questions'], 1):
        print(f"\nQ{i} ({question['type'].replace('_', ' ').title()}):")
        print(f"   {question['question']}")
        print(f"   Answer: {question['answer']}")
    
    # Show concept analysis
    print(f"\n🔍 Concept Analysis:")
    concepts = results['concept_analysis']['concepts']
    print(f"   Top 10 concepts extracted from CompTIA content:")
    for i, concept in enumerate(concepts[:10], 1):
        print(f"   {i:2d}. {concept['text']} ({concept['type']}, weight: {concept['weight']})")
    
    # Show relevant chunks
    print(f"\n📄 Relevant Google Notes Chunks:")
    chunks = results['relevant_chunks']
    for i, chunk_data in enumerate(chunks[:3], 1):  # Show top 3
        chunk = chunk_data['chunk']
        similarity = chunk_data['similarity']
        print(f"   Chunk {i} (Similarity: {similarity:.3f}):")
        print(f"   {chunk.get('text', '')[:150]}...")
    
    # Save results
    output_file = generator.save_results(results)
    if output_file:
        print(f"\n💾 Results saved to: {output_file}")
    
    print(f"\n🎉 Demo completed successfully!")
    print(f"💡 The system successfully:")
    print(f"   - Extracted {len(concepts)} concepts from CompTIA content")
    print(f"   - Found {len(chunks)} relevant Google notes chunks")
    print(f"   - Generated {len(results['questions'])} questions based on Google notes")
    print(f"   - All questions are grounded in the Google IT Support content")

def demo_with_custom_comptia_content():
    """Demo with custom CompTIA content"""
    print(f"\n" + "="*60)
    print("🔄 Demo with Custom CompTIA Content")
    print("="*60)
    
    # Custom CompTIA content about hardware troubleshooting
    custom_comptia = """
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
    """
    
    print(f"📚 Custom CompTIA Content:")
    print(f"Topic: Hardware Troubleshooting")
    print(f"Length: {len(custom_comptia)} characters")
    
    # Initialize generator
    generator = CrossReferenceQuestionGenerator()
    
    # Generate questions
    print(f"\n🔄 Generating questions for hardware troubleshooting...")
    results = generator.generate_cross_reference_questions(
        comptia_text=custom_comptia,
        num_questions=3,
        question_types=['multiple_choice', 'true_false', 'short_answer']
    )
    
    # Display results
    print(f"\n✅ Generation completed!")
    print(f"📊 Concepts extracted: {results['num_concepts_extracted']}")
    print(f"📄 Google notes chunks retrieved: {results['num_chunks_retrieved']}")
    print(f"❓ Questions generated: {len(results['questions'])}")
    
    # Show questions
    print(f"\n📋 Generated Questions:")
    for i, question in enumerate(results['questions'], 1):
        print(f"\nQ{i} ({question['type'].replace('_', ' ').title()}):")
        print(f"   {question['question']}")
        print(f"   Answer: {question['answer']}")
    
    # Save results
    output_file = f"demo_hardware_troubleshooting_{int(__import__('time').time())}.json"
    generator.save_results(results, output_file)
    print(f"\n💾 Results saved to: {output_file}")

def main():
    """Main demo function"""
    print("🚀 Cross-Reference Question Generator - Complete Demo")
    print("="*70)
    
    # Run main demo
    demo_cross_reference_system()
    
    # Run custom content demo
    demo_with_custom_comptia_content()
    
    print(f"\n🎉 All demos completed!")
    print(f"📁 Check the generated JSON files for detailed results")
    print(f"💡 You can now use this system with your own CompTIA content!")

if __name__ == "__main__":
    main()
