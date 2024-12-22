import gradio as gr
import io
import os
from file_manager import FileManager, FileType

# Initialize the file manager
file_manager = FileManager()

def process_file(file_obj):
    if file_obj is None:
        return "No file uploaded"
    
    try:
        # In Gradio 4.0+, file_obj is a NamedString with a name attribute pointing to the file path
        with open(file_obj.name, 'rb') as f:
            file_content = f.read()
        
        # Create a file-like object
        file_like = io.BytesIO(file_content)
        file_like.name = os.path.basename(file_obj.name)
        
        # Get file type before processing
        file_type = FileType.from_file(file_like)
        file_like.seek(0)  # Reset position after checking type
        
        # Process the file with status updates
        gr.Info(f"Processing {file_type.value} file: {file_like.name}")
        file_manager.upload_file(file_like)
        
        result_msg = f"✅ Successfully processed {file_type.value} file: {file_like.name}"
        gr.Info(result_msg)
        return result_msg
    except Exception as e:
        error_msg = f"❌ Error processing file: {str(e)}"
        gr.Error(error_msg)
        return error_msg

def process_url(url):
    if not url:
        return "No URL provided"
    
    try:
        gr.Info(f"Processing URL: {url}")
        # Convert URL to bytes
        url_bytes = io.BytesIO(url.encode('utf-8'))
        url_bytes.name = url  # Set name for file type detection
        
        # Process the URL
        file_manager.upload_file(url_bytes)
        result_msg = f"✅ Successfully processed weblink: {url}"
        gr.Info(result_msg)
        return result_msg
    except Exception as e:
        error_msg = f"❌ Error processing URL: {str(e)}"
        gr.Error(error_msg)
        return error_msg

# Create Gradio interface
with gr.Blocks(title="File Processing System", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📁 File Processing System")
    gr.Markdown("""
    ## Supported File Types:
    - 🎵 Audio Files (.mp3, .wav, .ogg, .m4a, .flac, .aac, .wma, .aiff)
                
    - 📄 PDF Files (.pdf)
    - 🖼️ Image Files (.jpg, .jpeg, .png, .gif, .bmp, .webp, .svg, .tiff)
    - 🔗 Web Links (URLs)
    """)
    
    with gr.Tab("📤 File Upload"):
        file_input = gr.File(
            label="Upload File",
            file_types=[
                ".mp3", ".wav", ".ogg", ".m4a", ".flac", ".aac", ".wma", ".aiff",
                ".pdf",
                ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".svg", ".tiff",
                ".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm", ".mpeg", ".mpg", ".3gp"
            ]
        )
        file_button = gr.Button("📤 Process File", variant="primary")
        file_output = gr.Textbox(label="Result", lines=4)
        file_button.click(
            process_file,
            inputs=[file_input],
            outputs=[file_output],
            show_progress=True
        )
    
    with gr.Tab("🔗 URL Processing"):
        url_input = gr.Textbox(
            label="Enter URL",
            placeholder="https://example.com",
            info="Enter a valid web URL to process"
        )
        url_button = gr.Button("🔗 Process URL", variant="primary")
        url_output = gr.Textbox(label="Result", lines=4)
        url_button.click(
            process_url,
            inputs=[url_input],
            outputs=[url_output],
            show_progress=True
        )

if __name__ == "__main__":
    demo.launch(share=True)
