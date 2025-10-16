import os
from pathlib import Path
from typing import List, Dict
import pdfplumber
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter  # ← CORRECTED
from dotenv import load_dotenv

load_dotenv()

class PDFProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file."""
        print(f"Extracting text from {pdf_path}...")
        text = ""
        
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            print(f"Total pages to process: {total_pages}")
            
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
                
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{total_pages} pages...")
        
        print(f"Extraction complete. Total characters: {len(text)}")
        return text
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """Split text into chunks with metadata."""
        print(f"Chunking text into {self.chunk_size} character chunks...")
        
        chunks = self.text_splitter.split_text(text)
        
        chunked_docs = []
        for i, chunk in enumerate(chunks):
            doc = {
                "text": chunk,
                "chunk_id": i,
                "metadata": metadata or {}
            }
            chunked_docs.append(doc)
        
        print(f"Created {len(chunked_docs)} chunks")
        return chunked_docs
    
    def process_main_guide(self, pdf_path: str) -> List[Dict]:
        """Process the main AWS certification guide."""
        text = self.extract_text_from_pdf(pdf_path)
        
        metadata = {
            "source": "aws_certification_guide",
            "doc_type": "study_guide",
            "filename": Path(pdf_path).name
        }
        
        return self.chunk_text(text, metadata)
    
    def process_practice_test(self, pdf_path: str) -> List[Dict]:
        """Process practice test PDFs."""
        text = self.extract_text_from_pdf(pdf_path)
        
        metadata = {
            "source": "practice_test",
            "doc_type": "questions",
            "filename": Path(pdf_path).name
        }
        
        return self.chunk_text(text, metadata)
    
    def save_chunks(self, chunks: List[Dict], output_path: str):
        """Save chunks to JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(chunks)} chunks to {output_path}")


def main():
    """Process all PDFs in the raw data directory."""
    processor = PDFProcessor(chunk_size=1000, chunk_overlap=200)
    
    raw_data_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    processed_dir.mkdir(exist_ok=True)
    
    # Check if PDFs exist
    pdf_files = list(raw_data_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"❌ No PDF files found in {raw_data_dir.absolute()}")
        print("Please add your PDF files to the data/raw/ directory")
        return
    
    print(f"Found {len(pdf_files)} PDF file(s) to process\n")
    
    all_chunks = []
    
    # Process all PDFs in the directory
    for pdf_file in pdf_files:
        print(f"\n{'='*60}")
        print(f"Processing: {pdf_file.name}")
        print(f"{'='*60}")
        
        try:
            # Determine document type and process accordingly
            if "practice" in pdf_file.name.lower() or "test" in pdf_file.name.lower():
                chunks = processor.process_practice_test(str(pdf_file))
            else:
                chunks = processor.process_main_guide(str(pdf_file))
            
            all_chunks.extend(chunks)
            
            # Save individual file chunks
            output_file = processed_dir / f"{pdf_file.stem}_chunks.json"
            processor.save_chunks(chunks, str(output_file))
            
        except Exception as e:
            print(f"❌ Error processing {pdf_file.name}: {e}")
            continue
    
    # Save all chunks together
    if all_chunks:
        all_chunks_file = processed_dir / "all_chunks.json"
        processor.save_chunks(all_chunks, str(all_chunks_file))
        
        print(f"\n{'='*60}")
        print(f"✅ Processing complete!")
        print(f"Total chunks created: {len(all_chunks)}")
        print(f"Files saved to: {processed_dir.absolute()}")
        print(f"{'='*60}")
    else:
        print("\n❌ No chunks were created. Check for errors above.")


if __name__ == "__main__":
    main() 
    
    