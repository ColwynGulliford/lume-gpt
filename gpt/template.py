BASIC_TEMPLATE=[
'setfile("beam", "gpt_particles.gdf");\n', 
'time=0.0;'
'\n', 
'auto_phase=0.0;\n', 
'space_charge=0.0;\n', 
'cathode=0.0;\n',
'space_charge_type=1.0;\n', 
'RadiusMax=0.04;\n', 
'GBacc=5.5;\n', 
'xacc=6.5;\n', 
'accuracy(GBacc, xacc);\n', 
'dtmin=1e-16;\n', 
'dtmax=1e-10;\n', 
'\n',  
'Alpha=1.0;\n', 
'Fn=0.5;\n', 
'verror=0.005;\n', 
'Nstd=5.0;\n', 
'tree_code_theta=1.0;\n', 
'tree_code_R=1e-06;\n', 
'if (space_charge == 1) {\n', 
'\n', 
'    if (space_charge_type == 1) {\n', 
'        if(cathode == 1) {\n'
'            spacecharge3Dmesh("Cathode", "MeshNfac", Alpha, "MeshAdapt", Fn, "SolverAcc", verror, "MeshBoxSize", Nstd);\n',
'        } else {\n'
'            spacecharge3Dmesh("MeshNfac", Alpha, "MeshAdapt", Fn, "SolverAcc", verror, "MeshBoxSize", Nstd);\n', 
'        }' 
'    }\n', 
'    if (space_charge_type == 2) {\n', 
'        setrmacrodist("beam","u",tree_code_R,0) ;\n', 
'        spacecharge3Dtree(tree_code_theta) ; \n', '    }\n', 
'}\n', 
'Ntout=50.0;\n', 
'tmax=10e-9;\n', 
'ZSTART=-0.005;\n', 
'ZSTOP=3;\n',
'zminmax("wcs", "I", ZSTART, ZSTOP);\n', 
'\n', 
'if(Ntout>0) {\n',
'    tout(time, tmax, tmax/Ntout);\n',
'}\n' 
'\n'
]

ZTRACK1_TEMPLATE=[
'setfile("beam", "gpt_particles.gdf");\n', 
'time=0.0;'
'\n', 
'RadiusMax=0.04;\n', 
'GBacc=5.5;\n', 
'xacc=6.5;\n', 
'accuracy(GBacc, xacc);\n', 
'dtmin=1e-16;\n', 
'dtmax=1e-10;\n', 
'\n',  
'tmax=1;\n', 
'ZSTART=-0.005;\n', 
'ZSTOP=3;\n',
'zminmax("wcs", "I", ZSTART, ZSTOP);\n', 
'\n'
]


def template(template_type='basic', filename='gpt.temp.in'):

    template_type = template_type.upper()+'_TEMPLATE'

    #with open()


def basic_template(filename='gpt.temp.in'):

    with open(filename, 'w') as fid:
        for line in BASIC_TEMPLATE:
            fid.write(line+'\n')

    return filename


def ztrack1_template(filename='gpt.temp.in'):

    with open(filename, 'w') as fid:
        for line in ZTRACK1_TEMPLATE:
            fid.write(line+'\n')

    return filename


