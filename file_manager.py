import io
import logging
import re
import os
from PIL import Image
from camel.loaders import ChunkrReader, Firecrawl
from camel.models import FishAudioModel
from enum import Enum
from dataclasses import dataclass
from PyPDF2 import PdfReader
from typing import Optional, BinaryIO, List, Tuple
from urllib.parse import urlparse

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

from camel.loaders import ChunkrReader, Firecrawl
from camel.models import FishAudioModel

# Assuming ImageProcessor is defined elsewhere as per your initial code

logger = logging.getLogger(__name__)

class FileType(Enum):
    AUDIO = "audio"
    PDF = "pdf"
    IMAGE = "image"
    WEBLINK = "weblink"
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
        self.pdf_reader = ChunkrReader()
        self.crawler = Firecrawl()
        self.image_processor = ImageProcessor(save_dir)

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
            task_id = self.pdf_reader.submit_task(pdf_file_path)
            logger.info(f"Submitted PDF processing task with ID: {task_id}")
            result = self.pdf_reader.get_task_output(task_id)
        except Exception as e:
            logger.warning(f"ChunkrReader failed, using PyPDF2 fallback: {e}")
            # Fallback to PyPDF2
            with open(pdf_file_path, 'rb') as pdf_file:
                pdf_reader = PdfReader(pdf_file)
                for page in pdf_reader.pages:
                    pdf_text += page.extract_text() + "\n"
            result = pdf_text

        output_path = os.path.join(pdf_file_path.replace('.pdf', '.txt'))
        with open(output_path, "w", encoding='utf-8') as f:
            f.write(result)
        logger.info(f"PDF processing complete, output saved to {output_path}")

    def _process_audio(self, file: File, saved_path: str):
        audio_file_path = saved_path
        audio_text = self.audio_model.speech_to_text(audio_file_path)
        with open(os.path.join(self.save_dir, f"{file.name}.txt"), "w") as f:
            f.write(audio_text)

    def upload_file(self, file: BinaryIO):
        file_type = FileType.from_file(file)

        if file_type in [FileType.AUDIO, FileType.PDF, FileType.IMAGE]:
            saved_path = self._save_file(file)
            logger.info(f"File saved to {saved_path}")
            if file_type == FileType.IMAGE:
                self.image_processor.process_image(File.from_upload(file))
            elif file_type == FileType.PDF:
                self._process_pdf(saved_path)
            elif file_type == FileType.AUDIO:
                self._process_audio(File.from_upload(file), saved_path)

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

        
