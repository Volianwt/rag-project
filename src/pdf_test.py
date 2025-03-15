import pdfplumber

pdf_path = "data/sample_docs/Unit3.pdf" 

with pdfplumber.open(pdf_path) as pdf:
    for i, page in enumerate(pdf.pages):
        text = page.extract_text()
        print(f"--- Page {i+1} ---")
        if text:
            print(text[:500])
        else:
            print("No text extracted from this page.")