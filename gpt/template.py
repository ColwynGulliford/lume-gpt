BASIC_TEMPLATE="""
setfile("beam", "gpt_particles.gdf");
time=0.0;

auto_phase=0.0;
space_charge=0.0;
cathode=0.0;
space_charge_type=1.0;
RadiusMax=0.04;
GBacc=5.5;
xacc=6.5;
accuracy(GBacc, xacc);
dtmin=1e-16;
dtmax=1e-10;
 
Alpha=1.0;
Fn=0.5;
verror=0.005;
Nstd=5.0;
tree_code_theta=1.0;
tree_code_R=1e-06;
if (space_charge == 1) {

    if (space_charge_type == 1) {
        if(cathode == 1) {
            spacecharge3Dmesh("Cathode", "MeshNfac", Alpha, "MeshAdapt", Fn, "SolverAcc", verror, "MeshBoxSize", Nstd);
        } else {
            spacecharge3Dmesh("MeshNfac", Alpha, "MeshAdapt", Fn, "SolverAcc", verror, "MeshBoxSize", Nstd);
        }
    }
    if (space_charge_type == 2) {
        setrmacrodist("beam","u",tree_code_R,0) ;
        spacecharge3Dtree(tree_code_theta) ;   }
        
    #if (space_charge_type == 3) {
    #    spacechargeP2Pgpu("3D", "SinglePrecision");
    #    }
}
Ntout=50.0;
tmax=10e-9;
ZSTART=-0.005;
ZSTOP=3;
zminmax("wcs", "I", ZSTART, ZSTOP);

if(Ntout>0) {
    tout(time, tmax, tmax/Ntout);
}
""".split('\n')

ZTRACK1_TEMPLATE="""
setfile("beam", "gpt_particles.gdf");
time=0.0;

RadiusMax=0.04;
GBacc=5.5;
xacc=6.5;
accuracy(GBacc, xacc);
dtmin=1e-16;
dtmax=1e-10;
 
tmax=1;
ZSTART=-0.005;
ZSTOP=3;
zminmax("wcs", "I", ZSTART, ZSTOP);
""".split('\n')


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


