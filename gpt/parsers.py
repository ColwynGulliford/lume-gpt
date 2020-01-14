import copy

# ------ Number parsing ------
def isfloat(value):
      try:
            float(value)
            return True
      except ValueError:
            return False

def parse_gpt_input_file(filePath, condense=False):
    """
    Parses GPT input file 
    """

    finput={}

    with open(filePath, 'r') as f:

        filestr=f.read()

        expressions = filestr.split(';')
        lines = []
        for ii,expression in enumerate(expressions):
            nlines = expression.strip().split('\n')
            for nline in nlines:
                lines.append(nline.strip())

        clean_lines=[]
    
        for line in lines:
            line = line.strip()
            line.replace('\n','')
            if(line!=''):
                tokens = line.split('#')
                if(tokens[0]!=''):
                    nline = tokens[0]
                    if(not (nline[-1]=='{' or nline[-1]=='}')):
                        nline = nline+';'

                    clean_lines.append(nline)
        #print(clean_lines)

    variables={}

    for ii,line in enumerate(clean_lines):
      
        tokens = line.split("=")

        if(len(tokens)==2 and isfloat(tokens[1][:-1].strip())):
 
            name = tokens[0].strip()
            value = float(tokens[1][:-1].strip())
            
            if(name not in variables.keys()):
                variables[name]=value #{"value":value,"index":ii}
                #print(name,value)
            else:
                print("Warning: multiple definitions of variable "+name+" on line "+str(ii)+".")

    finput['lines']=clean_lines
    finput['variables']=variables

    return finput


def write_gpt_input_file(finput,inputFile):

    print(inputFile)
    for var in finput["variables"].keys():

        value=finput["variables"][var]
        for index,line in enumerate(finput["lines"]):
            tokens = line.split("=")
            if(len(tokens)==2 and tokens[0].strip()==var):
                finput["lines"][index]=var+"="+str(value)+";"
                break

    with open(inputFile,'w') as f:

        for line in finput["lines"]:
            f.write(line+"\n")


