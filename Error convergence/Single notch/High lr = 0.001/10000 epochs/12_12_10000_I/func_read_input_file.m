function [SolverID,TangentID,ncoord,ndof,lc, ... 
            increment,inc_success_counter,min_iter,max_iter,max_accept_iter, ... 
            loadfactor,dlfactor,dlfactor_incr_threshold,increment_plot_threshold,loadfactor_plot_threshold, ...
            flaglf,countflaglf,incrflag,flagplot, ...
            ndomains,nprops,materialprops,alpha_val,beta_val,e_delta,dmax, ...
            nnodes,coords,nelem,maxnodes,connect,nelnodes,elident_vec,nfix,fixnodes] = func_read_input_file(infile) 

% ================== READ THE INPUT FILE WITH THE PARAMETERS ==============

% Reads the input text file and stores the variables accordingly
model_parameters_cellarray = textscan(infile,'%s');

cellnumber = 4;

% -------------------------------------------------------------------------
% Solver and Tangent IDs
SolverID    = str2double(model_parameters_cellarray{1}{cellnumber});
cellnumber  = cellnumber + 2;
TangentID   = str2double(model_parameters_cellarray{1}{cellnumber});

% -------------------------------------------------------------------------
% Number of coordinates and degrees of freedom
cellnumber  = cellnumber + 4;
ncoord      = str2double(model_parameters_cellarray{1}{cellnumber});
cellnumber  = cellnumber + 2;
ndof        = str2double(model_parameters_cellarray{1}{cellnumber});

% -------------------------------------------------------------------------
% Characteristic length
cellnumber  = cellnumber + 4;
lc          = str2double(model_parameters_cellarray{1}{cellnumber});

% -------------------------------------------------------------------------
% Adaptive loading parameters
cellnumber                  = cellnumber + 4;
increment                   = str2double(model_parameters_cellarray{1}{cellnumber});
cellnumber                  = cellnumber + 2;
inc_success_counter         = str2double(model_parameters_cellarray{1}{cellnumber});
cellnumber                  = cellnumber + 2;
min_iter                    = str2double(model_parameters_cellarray{1}{cellnumber});
cellnumber                  = cellnumber + 2;
max_iter                    = str2double(model_parameters_cellarray{1}{cellnumber});
cellnumber                  = cellnumber + 2;
max_accept_iter             = str2double(model_parameters_cellarray{1}{cellnumber});
cellnumber                  = cellnumber + 2;
loadfactor                  = str2double(model_parameters_cellarray{1}{cellnumber});
cellnumber                  = cellnumber + 2;
dlfactor                    = str2double(model_parameters_cellarray{1}{cellnumber});
cellnumber                  = cellnumber + 2;
dlfactor_incr_threshold     = str2double(model_parameters_cellarray{1}{cellnumber});
cellnumber                  = cellnumber + 2;
increment_plot_threshold    = str2double(model_parameters_cellarray{1}{cellnumber});
cellnumber                  = cellnumber + 2;
loadfactor_plot_threshold   = str2double(model_parameters_cellarray{1}{cellnumber});
cellnumber                  = cellnumber + 2;
flaglf                      = model_parameters_cellarray{1}{cellnumber};
cellnumber                  = cellnumber + 2;
countflaglf                 = str2double(model_parameters_cellarray{1}{cellnumber});
cellnumber                  = cellnumber + 2;
incrflag                    = str2double(model_parameters_cellarray{1}{cellnumber});
cellnumber                  = cellnumber + 2;
flagplot                    = str2double(model_parameters_cellarray{1}{cellnumber});

% -------------------------------------------------------------------------
% Number of domains
cellnumber  = cellnumber + 4;
ndomains    = str2double(model_parameters_cellarray{1}{cellnumber});

% -------------------------------------------------------------------------
% Material properties
cellnumber      = cellnumber + 4;
nprops          = str2double(model_parameters_cellarray{1}{cellnumber});
materialprops   = zeros(nprops,1);
for i = 1:nprops
    cellnumber = cellnumber + 2;  
    materialprops(i) = str2double(model_parameters_cellarray{1}{cellnumber});
end

% -------------------------------------------------------------------------
% Mazar's damage model parameters
cellnumber  = cellnumber + 4;
alpha_val   = str2double(model_parameters_cellarray{1}{cellnumber});
cellnumber  = cellnumber + 2;
beta_val    = str2double(model_parameters_cellarray{1}{cellnumber});
cellnumber  = cellnumber + 2;
e_delta     = str2double(model_parameters_cellarray{1}{cellnumber});
cellnumber  = cellnumber + 2;
dmax        = str2double(model_parameters_cellarray{1}{cellnumber});

% -------------------------------------------------------------------------
% Load mesh file
cellnumber = cellnumber + 3;
model_mesh_cellarray = textscan(fopen(model_parameters_cellarray{1}{cellnumber},'r'),'%s');

% ===================== READ THE INPUT FILE WITH THE MESH =================

% -------------------------------------------------------------------------
% Number and coordinates of nodes
cellnumber  = 2;
nnodes      = str2double(model_mesh_cellarray{1}{cellnumber});
cellnumber  = cellnumber + 2;
coords      = zeros(ncoord,nnodes);
for i = 1 : nnodes
    for j = 1 : ncoord
        coords(j,i) = str2double(model_mesh_cellarray{1}{cellnumber});
        cellnumber = cellnumber + 1;
    end
end

% -------------------------------------------------------------------------
% Number of elements and their connectivity
cellnumber  = cellnumber + 1;
nelem       = str2double(model_mesh_cellarray{1}{cellnumber});
cellnumber  = cellnumber + 2;
maxnodes    = str2double(model_mesh_cellarray{1}{cellnumber});
connect     = zeros(maxnodes,nelem);
nelnodes    = zeros(nelem,1);
elident_vec = zeros(nelem,1);
cellnumber  = cellnumber + 3;
for i = 1 : nelem
    cellnumber  = cellnumber + 1;
    elident_vec(i)  = str2double(model_mesh_cellarray{1}{cellnumber});
    cellnumber  = cellnumber + 1;
    nelnodes(i) = str2double(model_mesh_cellarray{1}{cellnumber});
    for j = 1 : nelnodes(i)
        cellnumber   = cellnumber + 1;
        connect(j,i) = str2double(model_mesh_cellarray{1}{cellnumber});
    end
end

% -------------------------------------------------------------------------
% Number of nodes with BCs and prescribed BCs
cellnumber  = cellnumber + 2;
nfix        = str2double(model_mesh_cellarray{1}{cellnumber});
cellnumber  = cellnumber + 3;
fixnodes    = zeros(3,nfix);
for i = 1 : nfix
    cellnumber      = cellnumber + 1;
    fixnodes(1,i)   = str2double(model_mesh_cellarray{1}{cellnumber});
    cellnumber      = cellnumber + 1;
    fixnodes(2,i)   = str2double(model_mesh_cellarray{1}{cellnumber});
    cellnumber      = cellnumber + 1;
    fixnodes(3,i)   = str2double(model_mesh_cellarray{1}{cellnumber});
end


    
end



