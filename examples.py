from docling.document_converter import DocumentConverter

source = "Camel AI with Ollama - Run Agents with Local Models Easily - Hands on_1.pdf"  # document per local path or URL
converter = DocumentConverter()
result = converter.convert(source)
print(result.document.export_to_markdown())
# output: ## Docling Technical Report [...]"