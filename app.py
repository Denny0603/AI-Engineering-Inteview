import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# -------------------
# Imports
# -------------------
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# -------------------
# Setup
# -------------------
load_dotenv()
app = Flask(__name__, static_folder='static', static_url_path='')

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not set in .env file")

PDF_PATH = "rag_data/about_me.pdf"

print("Loading PDF and creating vector store...")

# -------------------
# Load & index PDF once
# -------------------
try:
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()
    print(f"✅ Loaded {len(docs)} pages from PDF")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(docs)
    print(f"✅ Split into {len(chunks)} chunks")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("✅ Loaded embeddings model")

    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("✅ Created vector store")

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        groq_api_key=GROQ_API_KEY
    )
    print("✅ Initialized Groq LLM")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={
            "prompt": PromptTemplate(
                template="""Use the following context to answer the question. 
Keep your answer concise and to the point - aim for 2-3 sentences maximum.
If you don't know the answer, just say so briefly.

Context: {context}

Question: {question}

Concise Answer:""",
                input_variables=["context", "question"]
            )
        }
    )
    print("✅ QA chain ready!\n")

except Exception as e:
    print(f"❌ Error during setup: {e}")
    raise

# -------------------
# Email Function
# -------------------
def send_email_with_pdf(recipient_email, subject="Document from AI Assistant", body=""):
    """Send email with PDF attachment"""
    try:
        if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
            return False, "Email credentials not configured in .env file"

        # Create message
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = recipient_email
        msg['Subject'] = subject

        # Add body
        msg.attach(MIMEText(body, 'plain'))

        # Attach PDF
        with open(PDF_PATH, 'rb') as f:
            pdf_attachment = MIMEApplication(f.read(), _subtype='pdf')
            pdf_attachment.add_header('Content-Disposition', 'attachment', filename=os.path.basename(PDF_PATH))
            msg.attach(pdf_attachment)

        # Send email using Gmail SMTP
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)

        return True, "Email sent successfully!"

    except Exception as e:
        return False, f"Failed to send email: {str(e)}"

# -------------------
# Chat endpoint
# -------------------
@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_message = request.json.get("message")

        if not user_message:
            return jsonify({"response": "No message provided"}), 400

        print(f"Question: {user_message}")
        
        # Check if user wants to send email
        lower_msg = user_message.lower()
        if any(keyword in lower_msg for keyword in ["send email", "email", "send to", "email to"]):
            # Extract email address (simple regex-like approach)
            import re
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            emails = re.findall(email_pattern, user_message)
            
            if emails:
                recipient = emails[0]
                success, message = send_email_with_pdf(
                recipient,
                subject="Your Document",
                body="Attached is the requested document."
            )
                
                if success:
                    return jsonify({"response": f"✅ {message} The document has been sent to {recipient}."})
                else:
                    return jsonify({"response": f"❌ {message}"})
            else:
                return jsonify({"response": "Please provide a valid email address. For example: 'Send email to example@gmail.com'"})
        
        # Normal Q&A
        result = qa_chain({"query": user_message})
        print(f"Answer: {result['result']}\n")
        
        return jsonify({"response": result["result"]})
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"response": f"Error: {str(e)}"}), 500

# -------------------
# Frontend
# -------------------
@app.route("/")
def index():
    return app.send_static_file("index.html")

# if __name__ == "__main__":
#     print("🚀 Starting Flask server...")
#     print("📱 Open http://127.0.0.1:5000 in your browser\n")
#     app.run(debug=True)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
