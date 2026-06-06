#!/usr/bin/env python3
#olconvert.py
#
import ollama
import subprocess
import sys

def correct_text(raw_text_file):
    """
    Correct input text using Ollama.
    This function sends the input raw text to the Ollama API and returns the corrected equivalent.
    """
    try:
        # Prepare the prompt for Ollama
        prompt = (
            "Correct grammar and spelling of the following text "
            "Ensure the output text preserves the original formatting,"
            "handles errors gracefully, and follows best practices. "
            "Save this prompt at the top of the output text as a comment."
            "Return only the corrected text without any additional explanation or markdown formatting.\n\n"
            "Raw text:\n"
            "```bash\n"
            f"{raw_text_file}"
            "```\n\n"
            "Corrected text:"
        )
        
        # Use Ollama via the command line (assuming ollama is installed and running)
        # The model 'llama3' is commonly used; adjust if needed
        #result = subprocess.run(
        #    ["ollama", "run", "qwen3-coder-next"],
        #    input=prompt,
        #    capture_output=True,
        #    text=True,
        #    check=True
        #)
        
        #python_script = result.stdout.strip()
        response = ollama.generate(
            model='qwen3.6:35b',
            prompt=prompt,
            options={
              'seed': 12345,
              'temperature': 0  # Recommended for strict reproducibility
            }
        )
        #response = ollama.generate(model='qwen3-coder-next', prompt=prompt)       
        python_script = response['response']
        
        # Basic cleanup: remove potential markdown code fences
        if python_script.startswith("```python"):
            python_script = python_script.split("```python", 1)[1]
            python_script = python_script.split('\n', 1)[1]
        if python_script.startswith("```"):
            python_script = python_script.split("```", 1)[1]
            python_script = python_script.split('\n', 1)[1]
        if python_script.endswith("```"):
            python_script = python_script.rsplit("```", 1)[0]
          

        # Skip the f
        with open(output_python_path, 'w') as file:
             file.write(python_script)
    
        print(f"Conversion complete! Corected text saved to {output_python_path}")
        #return python_script.strip()
    
    except FileNotFoundError:
        print("Error: Ollama is not installed or not in PATH.", file=sys.stderr)
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error: Ollama command failed: {e.stderr}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return None

if __name__ == "__main__":
    # Read bash script from stdin or command line argument
    if len(sys.argv) > 2:
        with open(sys.argv[1], 'r') as f:
            raw_text = f.read()
        output_python_path=sys.argv[2]
        print(f"Converting {sys.argv[1]} into {output_python_path}")
    else:
        #bash_code = sys.stdin.read()
        #bash_code = ""
        print(f"Usage: {sys.argv[0]} raw_text_file corrected_text_file", file=sys.stderr)
        sys.exit(1)
        
    if not raw_text.strip():
        print("Error: No raw text file provided.", file=sys.stderr)
        sys.exit(1)
    
    corrected_text = correct_text(raw_text)
    
    if corrected_text:
        print(corrected_text)
    else:
        sys.exit(1)
