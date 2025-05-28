import logging
from pathlib import Path
from glob import glob

try :
    import ass
    import pandas as pd
    from pydub import AudioSegment
    from tqdm import tqdm
except ImportError :
    print("Lacking moduless, please run `pip install pydub pandas ass tqdm`")
    raise

_version = '1.2.0'

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

_logfile = Path(__file__).parent / 'debug.log'
# Logs that are written to file, level DEBUG
file_handler = logging.FileHandler(_logfile, mode='a', encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s')
file_handler.setFormatter(formatter)

# Logs that are printed to terminal, level INFO
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
formatter_simple = logging.Formatter('%(message)s')
stream_handler.setFormatter(formatter_simple)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)


def do_skip_line(line_text: str) :
    """Skip a line when condition is True

    Args:
        line_text (str): text of one line

    Returns:
        bool: True or False
    """
    if line_text.strip().startswith('*') :
        return True
    if line_text.strip().startswith('(') :
        return True
    return False


def process_audio_and_subtitles(ass_file_path: Path, output_dir: Path, separated_audio_folder: bool) -> None :
    """Process audio and subtitles."""
    
    # Ensure output directory exists
    if separated_audio_folder :
        separated_folder = output_dir / 'wavs' / ass_file_path.stem
    else :
        separated_folder = output_dir / 'wavs'
    separated_folder.mkdir(exist_ok=True, parents=True)

    # Load the subtitle file
    try :
        with open(ass_file_path, 'r', encoding='utf-8-sig') as f :
            ass_data = ass.parse(f)
    except ValueError as e :
        if 'encoding' in str(e) :
            logger.warning("Possibly encoding issue. Please run `dos2unix %s`" % (ass_file_path, ))
        raise
    logger.info("Subtitle loaded: %s" % (ass_file_path.name, ))

    # Parse the subtitle file
    # time in milliseconds(10^-3)
    subtitles = [(
        row.start.seconds * 1000 + row.start.microseconds / 1000,
        row.end.seconds * 1000 + row.end.microseconds / 1000,
        row.text
    ) for row in ass_data.events if not do_skip_line(row.text)]

    # Load the audio file
    audio_file = ass_file_path.with_name(ass_data.sections['Aegisub Project Garbage']['Audio File'])
    logger.info("Loading audio: %s" % (audio_file.name, ))
    if not audio_file.exists():
        audio_file = ass_file_path.with_suffix(".wav")
        
    if audio_file.exists() :
        audio = AudioSegment.from_file(audio_file, format="wav")
        logger.debug("Audio file loaded: %s" % (audio_file, ))
    else :
        logger.debug("Audio file not found: %s" % (audio_file, ))

    # Prepare data for CSV
    csv_data = []

    for i, (start_ms, end_ms, text) in tqdm(enumerate(subtitles, start=1), total=len(subtitles)) :
        # Extract the audio segment
        segment = audio[start_ms:end_ms]

        # Save the segment as a separate WAV file
        segment_file_name = f"{ass_file_path.stem}_segment_{i}.wav"
        segment_file_path = separated_folder / segment_file_name
        # Convert to 24bits
        segment.export(segment_file_path, parameters=["-c:a", "pcm_s24le"], format="wav")

        if separated_audio_folder :
            segment_file_name = separated_folder.name + '/' + segment_file_name
        # Append data for CSV
        csv_data.append({"File Name": segment_file_name, "Subtitle Text": text})

    # Create a DataFrame and save to CSV
    csv_file_path = output_dir / "metadata.csv"
    df = pd.DataFrame(csv_data)
    # No column title
    df.to_csv(csv_file_path, index=False, encoding='utf-8', sep='|', mode='a', header=False)


def main(args):
    ass_file_list = []
    for pattern in args.subtitles:
        if '*' in pattern or '?' in pattern :
            ass_file_list.extend(list(glob(pattern)))
        else :
            ass_file_list.append(pattern)
    
    output_dir = Path(args.output)
    for ass_file in ass_file_list :
        ass_file_path = Path(ass_file)
        process_audio_and_subtitles(ass_file_path, output_dir, args.keep_folder)


if __name__ == "__main__":
    from argparse import ArgumentParser
    # Arguments list
    parser = ArgumentParser(description='解析 ass 字幕并生成音频片段')
    parser.add_argument('-v', '--version', action='version', version=f"version {_version}", help='显示版本并退出')
    parser.add_argument('-o', '--output', type=str, nargs='?', default="output", help='输出目录')
    parser.add_argument('-k', '--keep-folder', action='store_true', help='每个音频文件保存至单独的目录')
    parser.add_argument('subtitles', metavar='ASS_FILE', type=str, nargs='+', help='ass 字幕文件')
    
    args = parser.parse_args()
    main(args)
