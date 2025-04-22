import argparse
import os
import random
import json
import numpy as np
import soundfile as sf
from pathlib import Path
import traceback
import torch

from dia.model import Dia


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the batch generation script."""
    parser = argparse.ArgumentParser(description="Generate audio for multiple tasks using the Dia model.")

    # --- Input File Argument ---
    parser.add_argument(
        "--input-file", type=str, required=True, help="Path to a JSON Lines (.jsonl) file containing generation tasks."
    )

    # --- Model Loading Arguments ---
    parser.add_argument(
        "--repo-id",
        type=str,
        default="nari-labs/Dia-1.6B",
        help="Hugging Face repository ID (e.g., nari-labs/Dia-1.6B).",
    )
    parser.add_argument(
        "--local-paths", action="store_true", help="Load model from local config and checkpoint files."
    )
    parser.add_argument(
        "--use-torch-compile",
        action="store_true",
        help="Enable torch.compile for potentially faster generation (after first run). Default is False."
    )
    parser.add_argument(
        "--config", type=str, help="Path to local config.json file (required if --local-paths is set)."
    )
    parser.add_argument(
        "--checkpoint", type=str, help="Path to local model checkpoint .pth file (required if --local-paths is set)."
    )
    # --- Default Generation Parameters (used if not specified in input file) ---
    gen_group = parser.add_argument_group("Default Generation Parameters")
    gen_group.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Default max tokens if not specified per task (uses config value if None).",
    )
    gen_group.add_argument(
        "--cfg-scale", type=float, default=3.0, help="Default CFG scale if not specified per task."
    )
    gen_group.add_argument(
        "--temperature", type=float, default=1.3, help="Default temperature if not specified per task."
    )
    gen_group.add_argument("--top-p", type=float, default=0.95, help="Default Top P if not specified per task.")

    # --- Infrastructure Arguments ---
    infra_group = parser.add_argument_group("Infrastructure")
    infra_group.add_argument("--seed", type=int, default=None, help="Global random seed for reproducibility for all tasks.")
    infra_group.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (e.g., 'cuda', 'cpu', default: auto).",
    )
    args = parser.parse_args()

    # Validation for local paths
    if args.local_paths:
        if not args.config:
            parser.error("--config is required when --local-paths is set.")
        if not args.checkpoint:
            parser.error("--checkpoint is required when --local-paths is set.")
        if not os.path.exists(args.config):
            parser.error(f"Config file not found: {args.config}")
        if not os.path.exists(args.checkpoint):
            parser.error(f"Checkpoint file not found: {args.checkpoint}")
    return args



def set_seed(seed: int):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cuDNN (if used)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def process_task(line:str, 
                 task_index:int, 
                 model:Dia, 
                 global_args:argparse.Namespace) -> bool:
    """
    Processes a single generation task defined by a line from the input file.
    Args:
        line: A string containing a single JSON object representing one task.
            The JSON object structure should be:
            {
                "text": "The input text for generation. (Required)",
                "output_path": "path/to/save/output.wav (Required)",
                "max_tokens": 1500 (Optional, overrides global default),
                "cfg_scale": 3.5 (Optional, overrides global default),
                "temperature": 1.2 (Optional, overrides global default),
                "top_p": 0.9 (Optional, overrides global default),
                "audio_prompt": "path/to/prompt.wav" (Optional, path to audio prompt for voice cloning)
            }
            Example Lines:
            {"text": "This is the first dialogue segment.", "output_path": "output/audio_01.wav"}
            {"text": "[S1] Hello there! [S2] General Kenobi!", "output_path": "output/star_wars.wav", "cfg_scale": 4.0, "temperature": 1.1}
            {"text": "A third example using default settings.", "output_path": "output/audio_03.wav"}
            {"text": "[S1] Clone this voice. [S2] Okay.", "output_path": "cloned/clone_test.wav", "audio_prompt": "prompts/speaker_a.wav"}

        task_index: The 1-based index of the task (for logging).
        model: The loaded Dia model instance.
        global_args: Parsed command-line arguments containing global defaults.

    Returns:
        True if the task was processed successfully, False otherwise.
    """
    print(f"\n--- Processing Task {task_index} ---")
    line = line.strip()
    if not line:
        print("  Skipping empty line.")
        return False # Treat empty lines as skipped/failed for counting purposes

    try:
        task_data = json.loads(line)

        # Get required task info 
        task_text = task_data.get("text")
        task_output_path_str = task_data.get("output_path")

        if not task_text:
            print(f"  Warning: Skipping task {task_index} - Missing 'text' field.")
            return False
        if not task_output_path_str:
            print(f"  Warning: Skipping task {task_index} - Missing 'output_path' field.")
            return False

        task_output_path = Path(task_output_path_str)

        # Determine generation parameters for this task:
        # Use task-specific value if present, otherwise use global arg default
        # Note: We access defaults directly from global_args
        task_max_tokens = task_data.get("max_tokens", global_args.max_tokens)
        task_cfg_scale = task_data.get("cfg_scale", global_args.cfg_scale)
        task_temperature = task_data.get("temperature", global_args.temperature)
        task_top_p = task_data.get("top_p", global_args.top_p)
        # Handle audio_prompt specifically: get from task data, default is None
        task_audio_prompt = task_data.get("audio_prompt", None) # No global default needed

        print(f"  Text: '{task_text[:50]}...'")
        print(f"  Output Path: {task_output_path}")
        print(f"  Parameters: max_tokens={task_max_tokens}, cfg={task_cfg_scale}, temp={task_temperature}, top_p={task_top_p}, prompt={task_audio_prompt}")

        # Generate Audio
        print("  Generating audio...")
        output_audio = model.generate(
            text=task_text,
            audio_prompt_path=task_audio_prompt,
            max_tokens=task_max_tokens,
            cfg_scale=task_cfg_scale,
            temperature=task_temperature,
            top_p=task_top_p,
            use_torch_compile=global_args.use_torch_compile # Global compile flag
        )
        print("  Audio generation complete.")

        # Save Audio
        print(f"  Saving audio to {task_output_path}...")
        task_output_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        sample_rate = 44100 # Assuming constant sample rate
        sf.write(str(task_output_path), output_audio, sample_rate)
        print(f"  Audio successfully saved.")
        return True # Task succeeded

    except json.JSONDecodeError:
        print(f"  Error: Could not parse JSON on line for task {task_index}. Skipping.")
        return False
    except FileNotFoundError as e:
         print(f"  Error processing task {task_index}: File not found - {e}. Skipping.")
         return False
    except Exception as e:
        print(f"  Error processing task {task_index} ('{task_output_path_str}'): {e}")
        traceback.print_exc() # Print full traceback for unexpected errors
        print("  Skipping task due to error.")
        return False


def main():
    args = parse_arguments()

    # Set seed if provided
    if args.seed is not None:
        set_seed(args.seed)
        print(f"Using random seed: {args.seed}")

    # Determine device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load model (once)
    print("Loading model...")
    if args.local_paths:
        print(f"Loading from local paths: config='{args.config}', checkpoint='{args.checkpoint}'")
        try:
            model = Dia.from_local(args.config, args.checkpoint, device=device)
        except Exception as e:
            print(f"Error loading local model: {e}")
            exit(1)
    else:
        print(f"Loading from Hugging Face Hub: repo_id='{args.repo_id}'")
        try:
            model = Dia.from_pretrained(args.repo_id, device=device)
        except Exception as e:
            print(f"Error loading model from Hub: {e}")
            exit(1)
    print("Model loaded.")

    # --- Process Tasks from Input File ---
    print(f"Processing tasks from: {args.input_file}")
    total_tasks = 0
    success_count = 0
    with open(args.input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines() # Read all lines at once for easier counting

    total_tasks = len([line for line in lines if line.strip()]) # Count non-empty lines

    for i, line in enumerate(lines):
        # Pass necessary info to the processing function
        if process_task(line, i + 1, model, args):
             success_count += 1

    print(f"\n--- Batch Processing Complete ---")
    print(f"Total tasks found: {total_tasks}")
    print(f"Successfully completed: {success_count}")
    print(f"Failed/Skipped: {total_tasks - success_count}")



if __name__ == "__main__":
    main()
