"""
Web Interface for Cross-Reference Question Generator
Simple Flask app for easy interaction with the question generation system
"""

from flask import Flask, render_template, request, jsonify, send_file
import os
import json
import time
from improved_question_generator import ImprovedCrossReferenceQuestionGenerator

app = Flask(__name__)
generator = None # Initialized lazily

def initialize_generator():
    """Initialize the improved question generator"""
    global generator
    if generator is None:
        print("🔄 Initializing improved question generator...")
        generator = ImprovedCrossReferenceQuestionGenerator()
        
        # Load Google notes
        google_notes_file = "extracted_Google IT Support Professional Certificate Notes (1).json"
        if os.path.exists(google_notes_file):
            generator.load_google_notes(google_notes_file)
            print("✅ Improved generator initialized successfully")
        else:
            print("❌ Google notes file not found")
            generator = None
    return generator

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_questions():
    """Generate questions using improved generator"""
    try:
        # Get form data
        comptia_content = request.form.get('comptia_content', '')
        num_questions = int(request.form.get('num_questions', 5))
        question_types = request.form.getlist('question_types')
        
        if not comptia_content.strip():
            return jsonify({'error': 'Please provide CompTIA content'})
        
        if not question_types:
            question_types = ['multiple_choice', 'true_false', 'short_answer']
        
        # Initialize generator
        gen = initialize_generator()
        if not gen:
            return jsonify({'error': 'System not ready. Please ensure Google notes are loaded.'})
        
        # Generate questions with improved quality
        print(f"🔄 Generating {num_questions} questions with improved quality...")
        results = gen.generate_cross_reference_questions(
            comptia_text=comptia_content,
            num_questions=num_questions,
            question_types=question_types
        )
        
        if 'error' in results:
            return jsonify({'error': results['error']})
        
        # Save results
        timestamp = int(time.time())
        filename = f"web_generated_questions_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Return success response
        return jsonify({
            'success': True,
            'message': f'Successfully generated {len(results["questions"])} questions with improved quality!',
            'filename': filename,
            'questions': results['questions'],
            'stats': {
                'total_questions': len(results['questions']),
                'comptia_text_length': results.get('comptia_text_length', 0),
                'num_concepts_extracted': results.get('num_concepts_extracted', 0),
                'num_chunks_retrieved': results.get('num_chunks_retrieved', 0)
            }
        })
        
    except Exception as e:
        print(f"❌ Error generating questions: {e}")
        return jsonify({'error': f'Error generating questions: {str(e)}'})

@app.route('/download/<filename>')
def download_file(filename):
    """Download generated questions file"""
    try:
        return send_file(filename, as_attachment=True)
    except Exception as e:
        return jsonify({'error': f'File not found: {filename}'})

@app.route('/status')
def status():
    """Check system status"""
    try:
        gen = initialize_generator()
        if gen and gen.google_notes:
            return jsonify({
                'status': 'ready',
                'message': '✅ System ready with improved quality!',
                'google_notes_loaded': len(gen.google_notes) if gen.google_notes else 0
            })
        else:
            return jsonify({
                'status': 'not_ready',
                'message': '❌ System not ready. Please ensure Google notes are loaded.'
            })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'❌ System error: {str(e)}'
        })

@app.route('/sample-content')
def sample_content():
    """Provide sample CompTIA content"""
    sample_content = """
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

Security measures such as encryption, authentication, and access control 
are essential for protecting sensitive data and maintaining network 
integrity. Regular security audits and updates help ensure that systems 
remain protected against evolving threats.
    """
    return jsonify({'content': sample_content.strip()})

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Create index.html template
    index_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cross-Reference Question Generator - Improved Quality</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5em;
        }
        .status {
            text-align: center;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            font-weight: bold;
        }
        .status.ready {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .status.not-ready {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #555;
        }
        textarea, input, select {
            width: 100%;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 16px;
            transition: border-color 0.3s;
        }
        textarea:focus, input:focus, select:focus {
            outline: none;
            border-color: #667eea;
        }
        .checkbox-group {
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }
        .checkbox-item {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 30px;
            border: none;
            border-radius: 8px;
            font-size: 18px;
            cursor: pointer;
            transition: transform 0.2s;
            width: 100%;
        }
        button:hover {
            transform: translateY(-2px);
        }
        button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        .results {
            margin-top: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
            display: none;
        }
        .question {
            background: white;
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        .question-type {
            font-weight: bold;
            color: #667eea;
            text-transform: uppercase;
            font-size: 0.9em;
        }
        .loading {
            text-align: center;
            padding: 20px;
            display: none;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .stats {
            background: #e9ecef;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .stats h3 {
            margin-top: 0;
            color: #495057;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        .stat-item {
            text-align: center;
            padding: 10px;
            background: white;
            border-radius: 5px;
        }
        .stat-value {
            font-size: 1.5em;
            font-weight: bold;
            color: #667eea;
        }
        .stat-label {
            font-size: 0.9em;
            color: #6c757d;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Cross-Reference Question Generator</h1>
        <h2 style="text-align: center; color: #667eea; margin-bottom: 30px;">Improved Quality Edition</h2>
        
        <div id="status" class="status">
            Checking system status...
        </div>
        
        <form id="questionForm">
            <div class="form-group">
                <label for="comptiaContent">CompTIA Content:</label>
                <textarea id="comptiaContent" name="comptia_content" rows="8" placeholder="Paste your CompTIA textbook content here..."></textarea>
                <button type="button" onclick="loadSampleContent()" style="width: auto; margin-top: 10px; background: #28a745;">📝 Load Sample Content</button>
            </div>
            
            <div class="form-group">
                <label for="numQuestions">Number of Questions:</label>
                <input type="number" id="numQuestions" name="num_questions" value="5" min="1" max="20">
            </div>
            
            <div class="form-group">
                <label>Question Types:</label>
                <div class="checkbox-group">
                    <div class="checkbox-item">
                        <input type="checkbox" id="multiple_choice" name="question_types" value="multiple_choice" checked>
                        <label for="multiple_choice">Multiple Choice</label>
                    </div>
                    <div class="checkbox-item">
                        <input type="checkbox" id="true_false" name="question_types" value="true_false" checked>
                        <label for="true_false">True/False</label>
                    </div>
                    <div class="checkbox-item">
                        <input type="checkbox" id="short_answer" name="question_types" value="short_answer" checked>
                        <label for="short_answer">Short Answer</label>
                    </div>
                </div>
            </div>
            
            <button type="submit" id="generateBtn">🎯 Generate Questions (Improved Quality)</button>
        </form>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>🔄 Generating high-quality questions...</p>
        </div>
        
        <div class="results" id="results">
            <h3>📊 Generation Statistics</h3>
            <div class="stats">
                <div class="stats-grid" id="statsGrid">
                    <!-- Stats will be populated here -->
                </div>
            </div>
            
            <h3>🎯 Generated Questions</h3>
            <div id="questionsList">
                <!-- Questions will be populated here -->
            </div>
            
            <div style="text-align: center; margin-top: 20px;">
                <button onclick="downloadResults()" style="width: auto; background: #17a2b8;">📥 Download Results</button>
            </div>
        </div>
    </div>

    <script>
        let currentFilename = null;
        
        // Check system status on page load
        window.onload = function() {
            checkStatus();
        };
        
        async function checkStatus() {
            try {
                const response = await fetch('/status');
                const data = await response.json();
                
                const statusDiv = document.getElementById('status');
                statusDiv.textContent = data.message;
                statusDiv.className = 'status ' + data.status;
                
                if (data.status === 'ready') {
                    document.getElementById('generateBtn').disabled = false;
                } else {
                    document.getElementById('generateBtn').disabled = true;
                }
            } catch (error) {
                console.error('Error checking status:', error);
                document.getElementById('status').textContent = '❌ Error checking system status';
                document.getElementById('status').className = 'status not-ready';
            }
        }
        
        async function loadSampleContent() {
            try {
                const response = await fetch('/sample-content');
                const data = await response.json();
                document.getElementById('comptiaContent').value = data.content;
            } catch (error) {
                console.error('Error loading sample content:', error);
                alert('Error loading sample content');
            }
        }
        
        document.getElementById('questionForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const generateBtn = document.getElementById('generateBtn');
            const loading = document.getElementById('loading');
            const results = document.getElementById('results');
            
            // Show loading
            generateBtn.disabled = true;
            loading.style.display = 'block';
            results.style.display = 'none';
            
            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.success) {
                    currentFilename = data.filename;
                    displayResults(data);
                } else {
                    alert('Error: ' + data.error);
                }
            } catch (error) {
                console.error('Error:', error);
                alert('Error generating questions');
            } finally {
                generateBtn.disabled = false;
                loading.style.display = 'none';
            }
        });
        
        function displayResults(data) {
            const results = document.getElementById('results');
            const statsGrid = document.getElementById('statsGrid');
            const questionsList = document.getElementById('questionsList');
            
            // Display stats
            statsGrid.innerHTML = `
                <div class="stat-item">
                    <div class="stat-value">${data.stats.total_questions}</div>
                    <div class="stat-label">Questions Generated</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">${data.stats.comptia_text_length}</div>
                    <div class="stat-label">CompTIA Text Length</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">${data.stats.num_concepts_extracted}</div>
                    <div class="stat-label">Concepts Extracted</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">${data.stats.num_chunks_retrieved}</div>
                    <div class="stat-label">Google Notes Chunks</div>
                </div>
            `;
            
            // Display questions
            questionsList.innerHTML = '';
            data.questions.forEach((question, index) => {
                const questionDiv = document.createElement('div');
                questionDiv.className = 'question';
                questionDiv.innerHTML = `
                    <div class="question-type">${question.type.replace('_', ' ').toUpperCase()}</div>
                    <div style="margin: 10px 0;"><strong>Q${index + 1}:</strong> ${question.question}</div>
                    <div><strong>Answer:</strong> ${question.answer}</div>
                `;
                questionsList.appendChild(questionDiv);
            });
            
            results.style.display = 'block';
        }
        
        function downloadResults() {
            if (currentFilename) {
                window.open(`/download/${currentFilename}`, '_blank');
            } else {
                alert('No results to download');
            }
        }
    </script>
</body>
</html>"""
    
    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write(index_html)
    
    print("🚀 Starting improved Cross-Reference Question Generator web app...")
    print("📱 Web interface available at: http://localhost:5000")
    print("✨ Now featuring improved quality with 100% answer completeness!")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
