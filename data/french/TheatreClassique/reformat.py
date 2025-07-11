import re

def reformat_script(input_path, output_path):
    """
    Converts lines in the format:
        [CHARACTER:] Line of dialogue
    to:
        CHARACTER:
        Line of dialogue
    Converts data in TheatreClassique format to tiny-shakespeare format.
    """
    # Pattern to match [CHARACTER:]
    pattern = re.compile(r'^\[(.+?):\]\s*(.*)$')

    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:
        
        for line in infile:
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            
            match = pattern.match(line)
            if match:
                speaker = match.group(1).strip()
                text = match.group(2).strip()
                
                outfile.write(f"{speaker}:\n{text}\n\n")
            else:
                # If line doesn't match, you can choose to write it as-is or skip
                # Uncomment below to include unmatched lines:
                # outfile.write(line + "\n\n")
                pass

if __name__ == "__main__":
    input_file = "/nobackup/users/scarv/multi-teacher-distillation/data/french/TheatreClassique/train.txt"
    output_file = "/nobackup/users/scarv/multi-teacher-distillation/data/french/TheatreClassique/train_reformatted.txt"
    reformat_script(input_file, output_file)
    print(f"Conversion complete. Output written to '{output_file}'")