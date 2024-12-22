import logging
import re
import os
from PIL import Image
from camel.loaders import ChunkrReader, Firecrawl
from camel.models import FishAudioModel
from enum import Enum
from dataclasses import dataclass
from moviepy import VideoFileClip
from PyPDF2 import PdfReader
from typing import BinaryIO
from urllib.parse import urlparse
from video_to_pdf import video_to_slides, slides_to_pdf
from docling.document_converter import DocumentConverter
from camel.agents import ChatAgent
from camel.configs import QwenConfig
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IMAGE_TO_TEXT_SYSTEM_PROMPT = "You are a helpful assistant that can describe the content of an image."
IMAGE_TO_TEXT_USER_PROMPT = "Please describe the content of the image in detail."

import os
import logging
import re
from enum import Enum
from dataclasses import dataclass
from typing import BinaryIO
from urllib.parse import urlparse

from camel.loaders import Firecrawl
from camel.models import FishAudioModel

# Assuming ImageProcessor is defined elsewhere as per your initial code

logger = logging.getLogger(__name__)

class FileType(Enum):
    AUDIO = "audio"
    PDF = "pdf"
    IMAGE = "image"
    WEBLINK = "weblink"
    VIDEO = "video"
    UNKNOWN = "unknown"

    @classmethod
    def from_file(cls, file: BinaryIO) -> "FileType":
        # Safeguard for files without a 'name' attribute
        filename = getattr(file, 'name', '').lower()

        # Web link patterns
        web_patterns = [
            r'\.html?$',
            r'^https?://',
            r'^www\.',
            r'\.(com|org|net|edu|gov|mil|io|co|me|app|dev|ai|txt)(/|$)',
            r'^localhost',
            r':\d{2,5}',
            r'/api/',
            r'/v\d+/',
            r'\?.*=.*',
            r'#.*$'
        ]

        # Check for web links first
        if any(re.search(pattern, filename) for pattern in web_patterns):
            try:
                parsed = urlparse(filename)
                if parsed.scheme or parsed.netloc:
                    return cls.WEBLINK
            except Exception:
                pass

        # Audio files
        if re.search(r'\.(mp3|wav|ogg|m4a|flac|aac|wma|aiff)$', filename):
            return cls.AUDIO

        # PDF files
        if filename.endswith('.pdf'):
            return cls.PDF

        # Image files
        if re.search(r'\.(jpg|jpeg|png|gif|bmp|webp|svg|tiff)$', filename):
            return cls.IMAGE
        
        # Video files
        if re.search(r'\.(mp4|avi|mkv|mov|wmv|flv|webm|mpeg|mpg|3gp)$', filename):
            return cls.VIDEO

        # Default to UNKNOWN
        return cls.UNKNOWN

@dataclass
class File:
    type: FileType
    name: str
    content: BinaryIO
    size: int | None

    @staticmethod
    def _get_binaryio_size_read(file: BinaryIO) -> int:
        current_pos = file.tell()
        file.seek(0, os.SEEK_END)
        size = file.tell()
        file.seek(current_pos)
        return size

    @classmethod
    def from_upload(cls, file_obj: BinaryIO | None) -> 'File':
        """Create File instance from uploaded file object"""
        if file_obj is None:
            raise ValueError("No file provided")
        
        # Get file type
        file_type = FileType.from_file(file_obj)

        # Get file name safely using basename to exclude directories
        name = os.path.basename(getattr(file_obj, 'name', 'unknown'))

        size = cls._get_binaryio_size_read(file_obj)

        return cls(
            type=file_type,
            name=name,
            content=file_obj,
            size=size
        )

class FileManager:
    def __init__(self, save_dir: str = "uploads"):
        self.save_dir = save_dir
        self.audio_model = FishAudioModel()
        self.pdf_converter = DocumentConverter()
        self.crawler = Firecrawl()
        self.image_processor = ImageProcessor(save_dir)
        self.video_processor = VideoProcessor(save_dir)

    def _save_file(self, file: BinaryIO):
        os.makedirs(self.save_dir, exist_ok=True)
        # Use os.path.basename to prevent directory traversal
        filename = os.path.basename(getattr(file, 'name', 'unknown'))
        filepath = os.path.join(self.save_dir, filename)
        
        # Prevent overwriting by appending a number if file exists
        base, ext = os.path.splitext(filepath)
        counter = 1
        while os.path.exists(filepath):
            filepath = f"{base}_{counter}{ext}"
            counter += 1

        with open(filepath, 'wb') as f:
            file.seek(0)
            f.write(file.read())
        return filepath
    
    def _process_pdf(self, pdf_file_path: str):
        pdf_text = ""
        try:
            converter = DocumentConverter()
            result = converter.convert(pdf_file_path)
        except Exception as e:
            logger.warning(f"ChunkrReader failed, using PyPDF2 fallback: {e}")
            # Fallback to PyPDF2
            with open(pdf_file_path, 'rb') as pdf_file:
                pdf_reader = PdfReader(pdf_file)
                for page in pdf_reader.pages:
                    pdf_text += page.extract_text() + "\n"
            result = pdf_text

        output_path = os.path.join(f"{pdf_file_path}.txt")
        with open(output_path, "w", encoding='utf-8') as f:
            f.write(result)
        logger.info(f"PDF processing complete, output saved to {output_path}")

    def _process_audio(self, saved_path: str):
        audio_file_path = saved_path
        audio_text = self.audio_model.speech_to_text(audio_file_path)
        with open(f"{audio_file_path}.txt", "w") as f:
            f.write(audio_text)

    def upload_file(self, file: BinaryIO):
        file_type = FileType.from_file(file)

        if file_type in [FileType.AUDIO, FileType.PDF, FileType.IMAGE, FileType.VIDEO]:
            saved_path = self._save_file(file)
            logger.info(f"File saved to {saved_path}")
            if file_type == FileType.IMAGE:
                self.image_processor.process_image(File.from_upload(file))
            elif file_type == FileType.PDF:
                self._process_pdf(saved_path)
            elif file_type == FileType.AUDIO:
                self._process_audio(saved_path)
            elif file_type == FileType.VIDEO:
                audio_path = self.video_processor.extract_audio(saved_path)
                self._process_audio(audio_path)
                pdf_path = self.video_processor.video_to_pdf(saved_path)
                self._process_pdf(pdf_path)       
        elif file_type == FileType.WEBLINK:
            # Assuming 'file' contains the URL as bytes
            try:
                url = file.read().decode('utf-8').strip()
                result = self.crawler.scrape(url)
                # Sanitize filename from URL
                parsed_url = urlparse(url)
                safe_filename = re.sub(r'\W+', '_', parsed_url.netloc + parsed_url.path)
                if not safe_filename:
                    safe_filename = 'weblink'
                filepath = os.path.join(self.save_dir, f"{safe_filename}.txt")
                with open(filepath, "w", encoding='utf-8') as f:
                    f.write(result['markdown'])
            except Exception as e:
                logger.error(f"Failed to process weblink: {e}")

    def concatenate_texts(self):
        files = [os.path.join(self.save_dir, file) for file in os.listdir(self.save_dir) if file.endswith('.txt')]
        texts = []
        for file in files:
            with open(file, "r") as f:
                texts.append(f'{file}: \n{f.read()}')
            concatenated_text = "\n".join(texts)
        with open(os.path.join(self.save_dir, "concatenated.txt"), "w") as f:
            f.write(concatenated_text)

class ImageProcessor:
    def __init__(self, save_dir: str):
        self.model = ModelFactory.create(
            model_platform=ModelPlatformType.QWEN,
            model_type=ModelType.QWEN_VL_PLUS,
            model_config_dict=QwenConfig(temperature=0.2).as_dict(),
        )
        self.save_dir = save_dir
        self.img_agent = ChatAgent(
            system_message=IMAGE_TO_TEXT_SYSTEM_PROMPT,
            model=self.model,
            output_language="English"  # Changed to match prompts
        )

    def process_image(self, image_file: File):
        # Open the image using PIL.Image
        try:
            pil_image = Image.open(image_file.content)
        except IOError:
            logger.error("Cannot convert binary data to an image.")
            return

        user_msg = BaseMessage.make_user_message(
            role_name="User",
            content=IMAGE_TO_TEXT_USER_PROMPT,
            image_list=[pil_image]
        )
            
        try:
            response = self.img_agent.step(user_msg)
            img_description = response.msgs[0].content

            # Ensure the filename is safe
            safe_filename = re.sub(r'\W+', '_', image_file.name)
            description_path = os.path.join(self.save_dir, f"{safe_filename}.txt")
            with open(description_path, "w", encoding='utf-8') as f:
                f.write(img_description)
            logger.info(f"Image description saved to {description_path}")
        except Exception as e:
            logger.error(f"Failed to process image: {e}")

class VideoProcessor:
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
    
    def extract_audio(self, saved_path: str):
        try:
            video_clip = VideoFileClip(saved_path)
            audio_clip = video_clip.audio
            audio_path = f"{saved_path}.mp3"
            audio_clip.write_audiofile(audio_path)
            logger.info(f"Audio extracted and saved to {audio_path}")
        except Exception as e:
            logger.error(f"Failed to extract audio: {e}")

        return audio_path
        
    def video_to_pdf(self, video_path: str) -> str:
        pdf_path = f"{video_path.replace('.mp4', '.pdf')}"
        try:    
            output_folder_screenshot_path, saved_files = video_to_slides(video_path)
            slides_to_pdf(video_path, output_folder_screenshot_path, saved_files)
            logger.info(f"PDF processing complete, output saved to {pdf_path}")
        except Exception as e:
            logger.error(f"Failed to process video: {e}")
        return pdf_path
    
    def merge_audio_and_pdf(self, audio_txt_path: str, pdf_txt_path: str):
        with open(audio_txt_path, "r") as audio_file:
            audio_text = audio_file.read()
        with open(pdf_txt_path, "r") as pdf_file:
            pdf_text = pdf_file.read()
        # remove the audio and pdf files
        os.remove(audio_txt_path)
        os.remove(pdf_txt_path)
        return f"Audio: \n{audio_text}\nPDF: \n{pdf_text}"
