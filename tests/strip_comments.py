import sys
import tokenize
import io
import os
import glob

def strip_comments_and_docstrings(source_code):
    io_obj = io.StringIO(source_code)
    out = ""
    prev_toktype = tokenize.INDENT
    last_lineno = -1
    last_col = 0
    
    try:
        for tok in tokenize.generate_tokens(io_obj.readline):
            token_type = tok[0]
            token_string = tok[1]
            start_line, start_col = tok[2]
            end_line, end_col = tok[3]
            
            if start_line > last_lineno:
                last_col = 0
            if start_col > last_col:
                out += " " * (start_col - last_col)
                
            if token_type == tokenize.COMMENT:
                pass
            elif token_type == tokenize.STRING:
                if prev_toktype != tokenize.INDENT and prev_toktype != tokenize.NEWLINE and start_col > 0:
                    out += token_string
            else:
                out += token_string
                
            prev_toktype = token_type
            last_col = end_col
            last_lineno = end_line
    except tokenize.TokenError:
        return source_code
        
    # Remove empty lines
    cleaned_lines = [line for line in out.splitlines() if line.strip()]
    return "\n".join(cleaned_lines) + "\n"

def main():
    engine_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'engine')
    for filepath in glob.glob(os.path.join(engine_dir, '**', '*.py'), recursive=True):
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()
        
        cleaned = strip_comments_and_docstrings(source)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(cleaned)
        print(f"Cleaned {filepath}")

if __name__ == "__main__":
    main()
