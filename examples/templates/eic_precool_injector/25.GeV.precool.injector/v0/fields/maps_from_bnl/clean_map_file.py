def n_tokens(line):
    return len(line.split())

def make_clean_txt_file(original_file):
    
    with open(original_file,'r') as fid:
        lines = fid.readlines()
        
    lengths = [n_tokens(lines[0])]
    for line in lines[1:]:
        lengths.append(n_tokens(line))        
    
    for ii in range(0,len(lengths)-1):
        print(lengths[ii], lengths[ii+1])
    