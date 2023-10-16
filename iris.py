import struct
import subprocess
import time
import logging
import traceback
from contextlib import contextmanager

import pvporcupine
import pvcobra
import pyaudio
from pydub import AudioSegment

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

CONFIG = {
    'chunk_size': 512,
    'sample_rate': 16000,
    'silence_threshold': 1.5,  # seconds
    'wake_word_paths': ['wake/hey_iris.ppn'],
    'format': pyaudio.paInt16,
    'channels': 1,
    'alert_sounds': {
        'wake': '/System/Library/Sounds/Ping.aiff',
        'success': '/System/Library/Sounds/Glass.aiff',
        'fail': '/System/Library/Sounds/Basso.aiff'
    }
}

def init_picovoice_modules(access_key):
    """Initialize Porcupine and Cobra."""
    try:
        porcupine = pvporcupine.create(access_key=access_key, keyword_paths=CONFIG['wake_word_paths'])
        cobra = pvcobra.create(access_key=access_key)
        return porcupine, cobra
    except Exception as e:
        logger.error("Failed to initialize Picovoice modules: ")
        logger.error(traceback.format_exc())
        raise

@contextmanager
def audio_stream(p):
    """Context manager for PyAudio to ensure resources are cleaned up."""
    stream = p.open(
        format=CONFIG['format'],
        channels=CONFIG['channels'],
        rate=CONFIG['sample_rate'],
        input=True,
        frames_per_buffer=CONFIG['chunk_size']
    )
    try:
        yield stream
    finally:
        stream.stop_stream()
        stream.close()

def get_next_audio_frame(stream):
    """Get the next audio frame from the stream, handle overflow."""
    frame_data = stream.read(CONFIG['chunk_size'], exception_on_overflow=False)
    return struct.unpack_from("h" * CONFIG['chunk_size'], frame_data)

def save_frames_as_mp3(frames, file_path="output.mp3"):
    """Save the audio frames as an MP3 file."""
    # Convert the list of frames into a single bytes object
    frames_bytes = b''.join(frames)

    # Calculate the number of bytes per frame: 2 bytes/sample * 1 channel
    bytes_per_frame = 2 * 1  # Modify as needed based on your audio format

    # Calculate if there are any leftover bytes and trim the frames_bytes if necessary
    remainder = len(frames_bytes) % bytes_per_frame
    if remainder != 0:
        logger.warning("Found an incomplete frame, trimming %s bytes.", remainder)
        frames_bytes = frames_bytes[:-remainder]

    # frames_bytes should have a length that is a multiple of bytes_per_frame
    try:
        audio_segment = AudioSegment(
            data=frames_bytes,
            sample_width=2,  # 2 bytes (16 bit)
            frame_rate=CONFIG['sample_rate'],
            channels=1  # mono
        )

        audio_segment.export(file_path, format="mp3")
        return True
    except Exception as e:
        logger.error("An error occurred while saving MP3: ")
        logger.error(traceback.format_exc())
        return False

def process_audio_stream(porcupine, cobra, stream):
    """Main loop to process audio stream, detect wake word, and record command."""
    frames = []
    wake_timestamp = None

    try:
        while True:
            frame = get_next_audio_frame(stream)
            keyword_index = porcupine.process(frame)

            if keyword_index >= 0:
                wake_timestamp = time.time()
                silence_timestamp = None
                frames = []
                logger.info(f"Keyword {keyword_index} detected, listening for command...")
                subprocess.Popen(["afplay", CONFIG['alert_sounds']['wake']])
                while True:
                    frame = get_next_audio_frame(stream)

                    frame_bytes = struct.pack("h" * len(frame), *frame)
                    frames.append(frame_bytes)
                    is_speech = cobra.process(frame) >= 0.5

                    if is_speech:
                        silence_timestamp = None # Speech detected, reset silence_timestamp
                    else:

                        if silence_timestamp is None: # First silence frame
                            silence_timestamp = time.time()
                        elif time.time() - silence_timestamp > CONFIG['silence_threshold']: # Silence for too long, reset
                            logging.info('Silence detected, saving command to mp3 and resetting...')
                            if save_frames_as_mp3(frames):
                                subprocess.Popen(["afplay", CONFIG['alert_sounds']['success']])
                            else:
                                subprocess.Popen(["afplay", CONFIG['alert_sounds']['fail']])
                            break
    except KeyboardInterrupt:
        logger.info("Stopping...")

def main():
    p = pyaudio.PyAudio()

    try:
        porcupine, cobra = init_picovoice_modules(PV_KEY)
        with audio_stream(p) as stream:
            process_audio_stream(porcupine, cobra, stream)
    except Exception as e:
        logger.error("An error occurred: ")
        logger.error(traceback.format_exc())
    finally:
        p.terminate()
        porcupine.delete()

if __name__ == "__main__":
    try:
        from secret import PV_KEY, OPENAI_KEY, HASS_KEY
    except ImportError:
        raise ImportError("Missing secret.py!")

    if not (PV_KEY and OPENAI_KEY and HASS_KEY):
        raise ValueError("Missing key(s) in secret.py!")

    main()
