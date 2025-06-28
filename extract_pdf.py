import PyPDF2
import sys

try:
    with open('Spungus trading system.pdf', 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages[:3]:  # First 3 pages
            text += page.extract_text() + "\n"
        print(text[:2000])  # First 2000 characters
except Exception as e:
    print(f"Error: {e}") 