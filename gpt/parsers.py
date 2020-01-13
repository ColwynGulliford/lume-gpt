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
        filestr=filestr.replace(' ','')
        lines = filestr.split('\n')

        clean_lines=[]
    
        for line in lines:
            line = line.strip()
            line.replace('\n','')
            if(line!=''):
                tokens = line.split('#')
                if(tokens[0]!=''):
                    clean_lines.append(tokens[0])

    variables={}

    for ii,line in enumerate(clean_lines):
      
        tokens = line.split("=")
        if(len(tokens)==2 and isfloat(tokens[1][:-1])):
 
            name = tokens[0]
            value = float(tokens[1][:-1])
            
            if(name not in variables.keys()):
                variables[name]={"value":value,"index":ii}
                #print(name,value)
            else:
                print("Warning: multiple definitions of variable "+name+" on line "+str(ii)+".")

    finput['lines']=clean_lines
    finput['variables']=variables

    return finput


def write_gpt_input_file(finput,inputFile):

    #print(inputFile)
    for var in finput["variables"].keys():

        value=finput["variables"][var]["value"]
        index=finput["variables"][var]["index"]
      
        finput["lines"][index]=var+"="+str(value)+";"

    with open(inputFile,'w') as f:

        for line in finput["lines"]:
            f.write(line+"\n")


