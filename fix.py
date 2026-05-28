import os
base_dir = 'quant_eurusd'
for file in os.listdir(base_dir):
    if file.endswith('_eurusd.py'):
        path = os.path.join(base_dir, file)
        with open(path, 'r', encoding='utf-8') as f:
            c = f.read()
        
        # O script 2 acabou escrevendo o \ e o n literalmente.
        c = c.replace('\\nif __name__ == "__main__":', '\nif __name__ == "__main__":')
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(c)
