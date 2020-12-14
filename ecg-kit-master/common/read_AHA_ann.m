%% Reads ECG annotations in AHA format
% Reads ECG annotations in AHA format. Implements the documentation
% available in the help of the application provided with the database.
% 
% Arguments:
%   + filename: recording to be read.
%   + start_sample: (opt) start sample to read. Default 1.
%   + end_sample: (opt) end sample to read. Default min(All recording, ECG block of 200 Mbytes)
% 
% Output:
%   + ann: annotations for the ECG recordings.
% 
% 
% See also read_AHA_format, read_AHA_header, read_ECG, ECGwrapper
% 
% Author: Mariano Llamedo Soria
% <matlab:web('mailto:llamedom@electron.frba.utn.edu.ar','-browser') (email)> 
% Version: 0.1 beta
% Birthdate: 17/12/2010
% Last update: 19/11/2014
% Copyright 2008-2015
% 
function [ ann last_sample] = read_AHA_ann( filename, start_sample, end_sample )

ann = [];
last_sample = [];

%No leer bloques mas grandes de 200 megabytes
MaxIOread = 200; %megabytes

if( nargin < 2 || isempty( start_sample ) )
    start_sample = 1;
else
    start_sample = max(1,start_sample);
end

fidECG = fopen( filename, 'r');

if( fidECG > 0 )

    fseek(fidECG, 0, 'eof');
    bytes_totales = ftell(fidECG);
    
    %esto es as� en la DB, si fuera necesario se podria cambiar.
    heasig.nsamp = (bytes_totales / 5);
    heasig.nsig = 2;

    if( nargin < 3 || isempty( end_sample ) )
        %Intento la lectura total por defecto
        samples2read = heasig.nsamp - (start_sample-1);
    else
        samples2read = min(heasig.nsamp, end_sample) - (start_sample-1);
    end
    
    if( (samples2read*heasig.nsig*2) > (MaxIOread * 1024^2) )
        samples2read = (MaxIOread * 1024^2) / heasig.nsig / 2;
        warning(['No es recomendable leer mas de ' num2str(MaxIOread) ' Mb. Realice varias lecturas.'])
    end
    
    try 
        
        fseek(fidECG, ((start_sample-1)*5)+4, 'bof');
        
        iAnnotations = fread(fidECG, samples2read, 'char', 4);
        
        fclose(fidECG);
        
    catch ME
        fclose(fidECG);
        rethrow(ME)
    end

    last_sample = size(iAnnotations,1) + start_sample - 1;
    
    bMask = iAnnotations ~= '.';

    ann.time = find(bMask);

    ann.anntyp = char(iAnnotations(bMask));

end
