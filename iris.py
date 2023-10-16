import pvporcupine
import pyaudio
import struct

try:
    from secret import PORCUPINE_ACCESS_KEY
except ImportError:
    raise ImportError("Missing Porcupine access key in secret.py!")

def get_next_audio_frame(stream):
    CHUNK_SIZE = 512
    return stream.read(CHUNK_SIZE, exception_on_overflow=False)

def main():
    CHUNK_SIZE = 512
    SAMPLE_RATE = 16000

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, input=True, frames_per_buffer=CHUNK_SIZE)

    handle = pvporcupine.create(access_key=PORCUPINE_ACCESS_KEY, keyword_paths=['wake/hey_iris.ppn'])

    try:
        while True:
            frame = get_next_audio_frame(stream)
            pcm = struct.unpack_from("h" * CHUNK_SIZE, frame)
            keyword_index = handle.process(pcm)
            if keyword_index >= 0:
                print(f"Keyword {keyword_index} detected!")
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        handle.delete()

if __name__ == "__main__":
    main()
