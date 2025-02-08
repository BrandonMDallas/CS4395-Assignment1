import math

def preprocess_line(line, lowercase=True):
    tokens = line.strip().split()
    
    if lowercase:
        tokens = [token.lower() for token in tokens]
        
    return tokens

def main():
    print("Main")
    
if __name__=="__main__":
    main()