# Counsel AI - Legal Ecosystem

Counsel AI is an advanced AI-powered legal assistance platform designed to simplify legal research, document analysis, and consultation. Built using state-of-the-art technologies such as OpenAI’s language models, LangChain, FAISS, and AWS services, the platform enables users to interact with legal information in a highly intuitive manner. Counsel AI supports legal professionals and individuals by providing accurate, context-aware responses to queries, eliminating the need for manual document reviews and extensive legal research.

## Features

- **AI-Powered Legal Chatbot**: Provides legal insights based on the Indian Penal Code and case laws.
- **Document Upload & Analysis**: Users can upload legal documents for AI-powered retrieval and analysis.
- **Embedded Legal Knowledge**: The system leverages FAISS-based embeddings for quick legal lookups.
- **File Management with AWS S3**: Secure document storage and retrieval.
- **User Authentication via AWS Cognito**: Ensures secure access and data protection.
- **Streamlit-Based Interactive UI**: Simple and intuitive user interface for seamless interaction.

## System Workflow

1. **User Authentication**: Secure login via AWS Cognito.
2. **Dashboard Access**: Users can choose between AI legal consultation and document analysis.
3. **Legal Chatbot**: Users enter queries, and AI retrieves relevant laws and case studies.
4. **Document Analysis**: Uploaded documents are processed for legal insights.
5. **File Storage & Management**: All user documents are securely stored in AWS S3.
6. **User Continues Interaction or Exits**.

## Installation

### Prerequisites
- Python 3.9+
- Virtual Environment (Recommended)
- OpenAI API Key
- AWS S3 and Cognito Configurations

### Setup
```sh
# Clone the repository
git clone https://github.com/your-username/Counsel-AI.git
cd Counsel-AI

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

```

### Environment Variables
Create a `.env` file and add the following:
```ini
OPENAI_API_KEY=your-api-key
AWS_ACCESS_KEY=your-aws-access-key
AWS_SECRET_KEY=your-aws-secret-key
AWS_S3_BUCKET=your-s3-bucket
AWS_COGNITO_USER_POOL_ID=your-cognito-pool-id
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

